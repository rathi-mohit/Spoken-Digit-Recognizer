#ifndef MFCC_HPP
#define MFCC_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

class MFCCExtractor {
public:
    // Default configuration for standard 16kHz speech recognition
    int num_filters = 26;
    int num_ceps = 13;
    int fft_size = 512;
    double frame_size_ms = 25.0;
    double frame_stride_ms = 10.0;
    double pre_emphasis_coeff = 0.97;

    // Main extraction function
    // Returns an (N_frames x 39) matrix of MFCCs + Deltas + Delta-Deltas
    Eigen::MatrixXd extract(const std::string& filepath) const;

private:
    struct WavData {
        std::vector<double> samples;
        int sample_rate;
    };

    WavData read_wav(const std::string& filepath) const;
    Eigen::MatrixXd create_mel_filterbank(double sample_rate) const;
    Eigen::VectorXd compute_dct(const Eigen::VectorXd& filter_energies) const;
};

#endif // MFCC_HPP