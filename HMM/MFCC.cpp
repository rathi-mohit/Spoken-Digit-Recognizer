#include "MFCC.hpp"
#include "pocketfft_hdronly.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Standard 16-bit PCM WAV Header
struct WavHeader {
    char riff[4]; int32_t fileSize; char wave[4]; char fmt[4]; int32_t fmtSize;
    int16_t audioFormat; int16_t numChannels; int32_t sampleRate; int32_t byteRate;
    int16_t blockAlign; int16_t bitsPerSample; char data[4]; int32_t dataSize;
};

MFCCExtractor::WavData MFCCExtractor::read_wav(const std::string& filepath) const {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open WAV file: " << filepath << "\n";
        return {};
    }

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));

    if (header.bitsPerSample != 16 || header.numChannels != 1) {
        std::cerr << "Only 16-bit Mono WAV supported: " << filepath << "\n";
        return {};
    }

    int num_samples = header.dataSize / sizeof(int16_t);
    std::vector<int16_t> raw(num_samples);
    file.read(reinterpret_cast<char*>(raw.data()), header.dataSize);

    std::vector<double> normalized(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        normalized[i] = raw[i] / 32768.0;
    }

    return {normalized, header.sampleRate};
}

Eigen::MatrixXd MFCCExtractor::create_mel_filterbank(double sample_rate) const {
    auto hz_to_mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
    auto mel_to_hz = [](double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); };

    double low_mel = hz_to_mel(300.0);
    double high_mel = hz_to_mel(sample_rate / 2.0);
    
    Eigen::VectorXd mel_points = Eigen::VectorXd::LinSpaced(num_filters + 2, low_mel, high_mel);
    std::vector<int> bins(num_filters + 2);
    for (int i = 0; i < num_filters + 2; ++i) {
        bins[i] = std::floor((fft_size + 1) * mel_to_hz(mel_points(i)) / sample_rate);
    }

    int num_hz_bins = fft_size / 2 + 1;
    Eigen::MatrixXd fbank = Eigen::MatrixXd::Zero(num_filters, num_hz_bins);

    for (int m = 1; m <= num_filters; ++m) {
        for (int k = bins[m - 1]; k < bins[m]; ++k) {
            if (k < num_hz_bins) fbank(m - 1, k) = (double)(k - bins[m - 1]) / (bins[m] - bins[m - 1]);
        }
        for (int k = bins[m]; k < bins[m + 1]; ++k) {
            if (k < num_hz_bins) fbank(m - 1, k) = (double)(bins[m + 1] - k) / (bins[m + 1] - bins[m]);
        }
    }
    return fbank;
}

Eigen::VectorXd MFCCExtractor::compute_dct(const Eigen::VectorXd& filter_energies) const {
    Eigen::VectorXd mfcc = Eigen::VectorXd::Zero(num_ceps);
    for (int i = 0; i < num_ceps; ++i) {
        double sum = 0.0;
        for (int j = 0; j < num_filters; ++j) {
            sum += filter_energies(j) * std::cos(M_PI * i * (j + 0.5) / num_filters);
        }
        mfcc(i) = sum;
    }
    return mfcc;
}

Eigen::MatrixXd MFCCExtractor::extract(const std::string& filepath) const {
    WavData audio = read_wav(filepath);
    if (audio.samples.empty()) return Eigen::MatrixXd(0, 0);

    // 1. Pre-emphasis Filter
    for (int i = audio.samples.size() - 1; i > 0; --i) {
        audio.samples[i] -= pre_emphasis_coeff * audio.samples[i - 1];
    }

    int frame_length = std::round((frame_size_ms / 1000.0) * audio.sample_rate);
    int frame_step = std::round((frame_stride_ms / 1000.0) * audio.sample_rate);
    
    if (audio.samples.size() < frame_length) return Eigen::MatrixXd(0, 0);
    int num_frames = 1 + (audio.samples.size() - frame_length) / frame_step;

    Eigen::MatrixXd fbank = create_mel_filterbank(audio.sample_rate);
    Eigen::MatrixXd mfccs(num_frames, num_ceps);

    // Hamming Window
    Eigen::VectorXd window(frame_length);
    for (int i = 0; i < frame_length; ++i) {
        window(i) = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (frame_length - 1));
    }

    // Setup PocketFFT parameters for Real-to-Complex 1D Transform
    pocketfft::shape_t shape_in { static_cast<size_t>(fft_size) };
    pocketfft::stride_t stride_in { sizeof(double) };
    pocketfft::stride_t stride_out { sizeof(std::complex<double>) };
    int num_hz_bins = fft_size / 2 + 1;
    
    std::vector<double> fft_in(fft_size, 0.0);
    std::vector<std::complex<double>> fft_out(num_hz_bins);

    for (int f = 0; f < num_frames; ++f) {
        int start = f * frame_step;
        
        // Windowing & zero-padding to fft_size
        std::fill(fft_in.begin(), fft_in.end(), 0.0);
        for (int i = 0; i < frame_length; ++i) {
            fft_in[i] = audio.samples[start + i] * window(i);
        }

        // Execute PocketFFT
        pocketfft::r2c(shape_in, stride_in, stride_out, 0, true, fft_in.data(), fft_out.data(), 1.0);

        // Power Spectrum
        Eigen::VectorXd power_spec(num_hz_bins);
        for (int i = 0; i < num_hz_bins; ++i) {
            double mag = std::abs(fft_out[i]);
            power_spec(i) = (mag * mag) / fft_size;
        }

        // Apply Mel-Filterbank and Log
        Eigen::VectorXd mel_energies = fbank * power_spec;
        for (int i = 0; i < mel_energies.size(); ++i) {
            mel_energies(i) = std::log(std::max(mel_energies(i), 1e-10));
        }

        // DCT to get MFCC
        mfccs.row(f) = compute_dct(mel_energies).transpose();
    }

    // Calculate Deltas and Delta-Deltas to hit 39 Dimensions
    Eigen::MatrixXd full_features(num_frames, 39);
    for (int f = 0; f < num_frames; ++f) {
        full_features.block(f, 0, 1, num_ceps) = mfccs.row(f);
        
        int prev = std::max(0, f - 1);
        int next = std::min(num_frames - 1, f + 1);
        Eigen::RowVectorXd delta = (mfccs.row(next) - mfccs.row(prev)) / 2.0;
        full_features.block(f, 13, 1, 13) = delta;

        int prev2 = std::max(0, f - 2);
        int next2 = std::min(num_frames - 1, f + 2);
        Eigen::RowVectorXd d_next = (mfccs.row(next2) - mfccs.row(f)) / 2.0;
        Eigen::RowVectorXd d_prev = (mfccs.row(f) - mfccs.row(prev2)) / 2.0;
        full_features.block(f, 26, 1, 13) = (d_next - d_prev) / 2.0;
    }
    Eigen::RowVectorXd mean =
    full_features.colwise().mean();

Eigen::RowVectorXd std =
(
 (full_features.rowwise()-mean)
 .array()
 .square()
 .colwise()
 .mean()
).sqrt();

full_features =
(
 full_features.rowwise()-mean
).array().rowwise()
/
(std.array()+1e-8);

return full_features;
}