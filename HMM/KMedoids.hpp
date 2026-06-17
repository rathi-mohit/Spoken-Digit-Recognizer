#ifndef KMEDOIDS_HPP
#define KMEDOIDS_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <stdexcept>

class KMedoids {
public:
    int n_clusters;
    int max_iter;
    int random_state;
    Eigen::MatrixXd medoids;
    std::vector<int> labels_;

    KMedoids(int n_clusters, int max_iter = 100, int random_state = 42)
        : n_clusters(n_clusters), max_iter(max_iter), random_state(random_state) {
        if (n_clusters <= 0) {
            throw std::invalid_argument("Number of clusters must be positive.");
        }
    }

    void fit(const Eigen::MatrixXd& X) {
        int n_samples = X.rows();
        if (n_samples < n_clusters) {
            throw std::invalid_argument("Number of samples less than number of clusters.");
        }

        std::mt19937 gen(random_state);
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        medoids = Eigen::MatrixXd(n_clusters, X.cols());
        for (int i = 0; i < n_clusters; ++i) {
            medoids.row(i) = X.row(indices[i]);
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<std::vector<int>> clusters = assign_clusters(X);
            Eigen::MatrixXd new_medoids = update_medoids(X, clusters);

            if (medoids.isApprox(new_medoids, 1e-8)) {
                break;
            }
            medoids = new_medoids;
        }

        labels_ = predict(X);
    }

    std::vector<int> predict(const Eigen::MatrixXd& X) const {
        if (medoids.rows() == 0) {
            throw std::runtime_error("Model not fitted yet.");
        }
        int n_samples = X.rows();
        std::vector<int> labels(n_samples);

        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            Eigen::RowVectorXd point = X.row(i);
            double min_dist = std::numeric_limits<double>::infinity();
            int best_cluster = -1;

            for (int j = 0; j < n_clusters; ++j) {
                // Manhattan (L1) Distance
                double dist = (point - medoids.row(j)).cwiseAbs().sum();
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels[i] = best_cluster;
        }
        return labels;
    }

private:
    std::vector<std::vector<int>> assign_clusters(const Eigen::MatrixXd& X) const {
        int n_samples = X.rows();
        std::vector<std::vector<int>> clusters(n_clusters);
        
        // Use an array of vectors to safely accumulate thread-local clusters
        // Avoids race conditions on push_back during parallel execution
        #pragma omp parallel
        {
            std::vector<std::vector<int>> local_clusters(n_clusters);
            
            #pragma omp for nowait
            for (int i = 0; i < n_samples; ++i) {
                Eigen::RowVectorXd point = X.row(i);
                double min_dist = std::numeric_limits<double>::infinity();
                int best_cluster = -1;

                for (int j = 0; j < n_clusters; ++j) {
                    double dist = (point - medoids.row(j)).cwiseAbs().sum();
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                local_clusters[best_cluster].push_back(i);
            }

            #pragma omp critical
            {
                for (int j = 0; j < n_clusters; ++j) {
                    clusters[j].insert(clusters[j].end(), local_clusters[j].begin(), local_clusters[j].end());
                }
            }
        }
        return clusters;
    }

    Eigen::MatrixXd update_medoids(const Eigen::MatrixXd& X, const std::vector<std::vector<int>>& clusters) const {
        Eigen::MatrixXd new_medoids(n_clusters, X.cols());

        #pragma omp parallel for
        for (int k = 0; k < n_clusters; ++k) {
            const auto& points_idx = clusters[k];
            if (points_idx.empty()) {
                new_medoids.row(k) = medoids.row(k); // Fallback if empty
                continue;
            }

            double best_cost = std::numeric_limits<double>::infinity();
            int best_idx = -1;

            for (int candidate_idx : points_idx) {
                Eigen::RowVectorXd candidate = X.row(candidate_idx);
                double cost = 0.0;
                for (int other_idx : points_idx) {
                    cost += (candidate - X.row(other_idx)).cwiseAbs().sum();
                }

                if (cost < best_cost) {
                    best_cost = cost;
                    best_idx = candidate_idx;
                }
            }
            new_medoids.row(k) = X.row(best_idx);
        }
        return new_medoids;
    }
};

#endif // KMEDOIDS_HPP