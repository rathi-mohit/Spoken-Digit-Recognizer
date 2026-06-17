#include "MFCC.hpp"
#include "KMedoids.hpp"
#include "HMM.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

namespace fs = std::filesystem;

const int N_STATES = 8;
const int N_CLUSTERS = 64;

MFCCExtractor mfcc_extractor;
KMedoids codebook(N_CLUSTERS);
std::map<std::string, HiddenMarkovModel> digit_hmms;


// Utility function to stack a vector of Eigen matrices vertically
Eigen::MatrixXd vstack(const std::vector<Eigen::MatrixXd>& matrices) {
    int total_rows = 0;
    int cols = matrices.empty() ? 0 : matrices[0].cols();
    for (const auto& mat : matrices) {
        total_rows += mat.rows();
    }
    
    Eigen::MatrixXd stacked(total_rows, cols);
    int current_row = 0;
    for (const auto& mat : matrices) {
        stacked.middleRows(current_row, mat.rows()) = mat;
        current_row += mat.rows();
    }
    return stacked;
}

void train_system(const std::string& dataset_path) {
    std::vector<std::string> all_files;
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.path().extension() == ".wav") {
            all_files.push_back(entry.path().filename().string());
        }
    }

    std::map<std::string, std::vector<Eigen::MatrixXd>> training_data;
    std::vector<Eigen::MatrixXd> all_frames;

    std::cout << "Extracting Features...\n";
    for (int d = 0; d < 10; ++d) {
        for (int s = 1; s <= 5; ++s) {
            std::string prefix = std::to_string(d) + "_speaker" + std::to_string(s) + "_";
            for (const auto& f : all_files) {
                if (f.find(prefix) == 0) {
                    Eigen::MatrixXd feat = mfcc_extractor.extract(dataset_path + "/" + f);
                    if (feat.rows() > 0) {
                        training_data[std::to_string(d)].push_back(feat);
                        all_frames.push_back(feat);
                    }
                }
            }
        }
    }

    std::cout << "Training Codebook (K-Medoids)...\n";
    Eigen::MatrixXd X_all = vstack(all_frames);
    
    // Subsample for clustering (e.g. ::10)
    Eigen::MatrixXd X_subsampled(X_all.rows() / 10, X_all.cols());
    for (int i = 0, j = 0; i < X_all.rows() && j < X_subsampled.rows(); i += 10, ++j) {
        X_subsampled.row(j) = X_all.row(i);
    }
    codebook.fit(X_subsampled);

    std::cout << "Training HMMs for Digits (Parallel)...\n";
    
    // Using OpenMP here since HMMs are completely independent
    #pragma omp parallel for
    for (int d = 0; d < 10; ++d) {
        std::string d_str = std::to_string(d);
        std::vector<std::vector<int>> quantized_sequences;
        
        for (const auto& file_feat : training_data[d_str]) {
            std::vector<int> indices = codebook.predict(file_feat);
            quantized_sequences.push_back(indices);
        }

        HiddenMarkovModel hmm(N_STATES, N_CLUSTERS);
        hmm.fit(quantized_sequences);
        
        #pragma omp critical
        {
            digit_hmms[d_str] = hmm;
            std::cout << "Finished training Digit " << d << "\n";
        }
    }
}

int infer(const std::string& filename) {
    Eigen::MatrixXd feat = mfcc_extractor.extract(filename);
    if (feat.rows() == 0) return -1;

    std::vector<int> obs = codebook.predict(feat);

    double best_score = -std::numeric_limits<double>::infinity();
    int best_digit = -1;

    for (const auto& [digit, model] : digit_hmms) {
        double score = model.score(obs);
        if (score > best_score) {
            best_score = score;
            best_digit = std::stoi(digit);
        }
    }

    return best_digit;
}

void calculate_metrics(const std::vector<int>& true_labels, const std::vector<int>& pred_labels, const std::string& dataset_name) {
    int total = true_labels.size();
    int correct = 0;
    Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(10, 10);
    
    for (size_t i = 0; i < true_labels.size(); ++i) {
        int t = true_labels[i];
        int p = pred_labels[i];
        if (t == p) correct++;
        cm(t, p)++;
    }
    
    double accuracy = (total > 0) ? (double)correct / total : 0.0;
    
    std::cout << "\n=== " << dataset_name << " Metrics ===\n";
    std::cout << "Overall Accuracy: " << (accuracy * 100.0) << "%\n";
    std::cout << "\nConfusion Matrix:\n" << cm << "\n";
    
    std::cout << "\nPer-Digit Accuracy:\n";
    for (int i = 0; i < 10; ++i) {
        int digit_total = cm.row(i).sum();
        int digit_correct = cm(i, i);
        double digit_acc = (digit_total > 0) ? (double)digit_correct / digit_total : 0.0;
        std::cout << "Digit " << i << ": " << (digit_acc * 100.0) << "%\n";
    }
}

int main() {
    std::string dataset_path = "../recordings";

    train_system(dataset_path);

    std::vector<std::string> all_files;
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.path().extension() == ".wav") {
            all_files.push_back(entry.path().filename().string());
        }
    }

    std::cout << "\nEvaluating Speakers 1-5 (Training Set)...\n";
    std::vector<int> train_true, train_pred;
    for (int d = 0; d < 10; ++d) {
        for (int s = 1; s <= 5; ++s) {
            std::string prefix = std::to_string(d) + "_speaker" + std::to_string(s) + "_";
            for (const auto& f : all_files) {
                if (f.find(prefix) == 0) {
                    int prediction = infer(dataset_path + "/" + f);
                    if (prediction != -1) {
                        train_true.push_back(d);
                        train_pred.push_back(prediction);
                    }
                }
            }
        }
    }
    calculate_metrics(train_true, train_pred, "Training (Speakers 1-5)");

    std::cout << "\nEvaluating Speaker 6 (Test Set)...\n";
    std::vector<int> test_true, test_pred;
    for (int d = 0; d < 10; ++d) {
        std::string prefix = std::to_string(d) + "_speaker6_";
        for (const auto& f : all_files) {
            if (f.find(prefix) == 0) {
                int prediction = infer(dataset_path + "/" + f);
                if (prediction != -1) {
                    test_true.push_back(d);
                    test_pred.push_back(prediction);
                }
            }
        }
    }
    calculate_metrics(test_true, test_pred, "Test (Speaker 6)");

    return 0;
}