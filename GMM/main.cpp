#include "GMM.hpp"
#include "MFCC.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// Global Scalers
Eigen::RowVectorXd GLOBAL_MEAN;
Eigen::RowVectorXd GLOBAL_STD;
MFCCExtractor mfcc_extractor;
std::map<std::string, GaussianMixtureModel> digit_models;


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

int infer(const std::string& filename) {
    Eigen::MatrixXd features = mfcc_extractor.extract(filename);
    if (features.rows() == 0) return -1;
    
    int num_frames = features.rows();
    
    // Apply Global Scaler
    features = (features.rowwise() - GLOBAL_MEAN).array().rowwise() / GLOBAL_STD.array();
    
    double best_score = -std::numeric_limits<double>::infinity();
    int predicted_digit = -1;
    
    for (const auto& [digit, model] : digit_models) {
        if (model.mu.size() > 0) {
            double score = model.score(features) / num_frames;
            if (score > best_score) {
                best_score = score;
                predicted_digit = std::stoi(digit);
            }
        }
    }
    return predicted_digit;
}

std::pair<std::vector<int>, std::vector<int>> evaluate_models(const std::string& dataset_path, const std::vector<int>& speaker_list) {
    std::vector<int> true_labels;
    std::vector<int> pred_labels;
    
    std::vector<std::string> all_files;
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.path().extension() == ".wav") {
            all_files.push_back(entry.path().filename().string());
        }
    }
    
    for (int digit = 0; digit < 10; ++digit) {
        std::cout << "Evaluating digit " << digit << "...\n";
        for (int speaker_id : speaker_list) {
            std::string prefix = std::to_string(digit) + "_speaker" + std::to_string(speaker_id) + "_";
            
            for (const auto& filename : all_files) {
                if (filename.find(prefix) == 0) {
                    std::string file_path = dataset_path + "/" + filename;
                    int pred = infer(file_path);
                    if (pred != -1) {
                        true_labels.push_back(digit);
                        pred_labels.push_back(pred);
                    }
                }
            }
        }
    }
    return {true_labels, pred_labels};
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
    std::cout << "Overall Accuracy: " << (accuracy * 100.0) << "% (" << correct << "/" << total << ")\n";
    
    std::cout << "\nConfusion Matrix (Row: True, Col: Predicted):\n";
    std::cout << cm << "\n";
    
    std::cout << "\nPer-Digit Accuracy:\n";
    for (int i = 0; i < 10; ++i) {
        int digit_total = cm.row(i).sum();
        int digit_correct = cm(i, i);
        double digit_acc = (digit_total > 0) ? (double)digit_correct / digit_total : 0.0;
        std::cout << "Digit " << i << ": " << (digit_acc * 100.0) << "%\n";
    }
}

void train_digit_models(const std::string& dataset_path) {
    std::map<std::string, std::vector<Eigen::MatrixXd>> raw_digit_features;
    std::vector<Eigen::MatrixXd> all_training_frames;
    
    std::vector<std::string> all_files;
    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.path().extension() == ".wav") {
            all_files.push_back(entry.path().filename().string());
        }
    }
    
    for (int digit = 0; digit < 10; ++digit) {
        digit_models[std::to_string(digit)] = GaussianMixtureModel(4, 100);
        for (int speaker_id = 1; speaker_id <= 5; ++speaker_id) {
            std::string prefix = std::to_string(digit) + "_speaker" + std::to_string(speaker_id) + "_";
            for (const auto& filename : all_files) {
                if (filename.find(prefix) == 0) {
                    std::string file_path = dataset_path + "/" + filename;
                    Eigen::MatrixXd features = mfcc_extractor.extract(file_path);
                    if (features.rows() > 0) {
                        raw_digit_features[std::to_string(digit)].push_back(features);
                        all_training_frames.push_back(features);
                    }
                }
            }
        }
    }
    
    // Calculate Global Mean and Std
    Eigen::MatrixXd massive_training_matrix = vstack(all_training_frames);
    GLOBAL_MEAN = massive_training_matrix.colwise().mean();
    GLOBAL_STD = ((massive_training_matrix.rowwise() - GLOBAL_MEAN).array().square().colwise().sum() / massive_training_matrix.rows()).sqrt() + 1e-10;
    
    // Normalize and Train
    for (int digit = 0; digit < 10; ++digit) {
        std::string d_str = std::to_string(digit);
        if (!raw_digit_features[d_str].empty()) {
            Eigen::MatrixXd X_raw = vstack(raw_digit_features[d_str]);
            
            // Apply global scaler
            Eigen::MatrixXd X_train = (X_raw.rowwise() - GLOBAL_MEAN).array().rowwise() / GLOBAL_STD.array();
            
            std::cout << "Training GMM for digit " << digit << " on " << X_train.rows() << " frames...\n";
            digit_models[d_str].fit(X_train);
        }
    }
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    std::string dataset_directory = "../recordings"; // Make sure path exists
    
    std::cout << "--- Evaluating Train Data ---\n";
    train_digit_models(dataset_directory);
    
    std::cout << "\n--- Evaluating Test Data ---\n";
    auto [train_true, train_pred] = evaluate_models(dataset_directory, {1, 2, 3, 4, 5});
    calculate_metrics(train_true, train_pred, "Train");
    
    std::cout << "\n--- Calculating Test Metrics ---\n";
    auto [test_true, test_pred] = evaluate_models(dataset_directory, {6});
    calculate_metrics(test_true, test_pred, "Test");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nTotal Execution Time: " << duration.count() << " ms\n";
    
    return 0;
}