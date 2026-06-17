#ifndef GMM_HPP
#define GMM_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>

class GaussianMixtureModel {
public:
    int n_components;
    int max_iter;
    double tol;
    int random_state;

    Eigen::VectorXd pi;
    Eigen::MatrixXd mu;
    std::vector<Eigen::VectorXd> sigma; // Storing diagonal variances
    std::vector<double> log_likelihood_history;

    GaussianMixtureModel(int n_comp = 2, int m_iter = 100, double t = 1e-6, int rs = 42)
        : n_components(n_comp), max_iter(m_iter), tol(t), random_state(rs) {}

    void fit(const Eigen::MatrixXd& X) {
        initialize_parameters(X);
        double prev_ll = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < max_iter; ++i) {
            Eigen::MatrixXd resp = e_step(X);
            m_step(X, resp);

            double curr_ll = compute_log_likelihood(X);
            log_likelihood_history.push_back(curr_ll);

            if (std::abs(curr_ll - prev_ll) < tol) {
                break;
            }
            prev_ll = curr_ll;
        }
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd resp = e_step(X);
        Eigen::VectorXi predictions(X.rows());
        
        #pragma omp parallel for
        for (int i = 0; i < X.rows(); ++i) {
            int max_idx;
            resp.row(i).maxCoeff(&max_idx);
            predictions(i) = max_idx;
        }
        return predictions;
    }

    double score(const Eigen::MatrixXd& X) const {
        if (mu.size() == 0) return -std::numeric_limits<double>::infinity();
        return compute_log_likelihood(X);
    }

private:
    void initialize_parameters(const Eigen::MatrixXd& X) {
        int n_samples = X.rows();
        int n_features = X.cols();

        pi = Eigen::VectorXd::Constant(n_components, 1.0 / n_components);
        mu = Eigen::MatrixXd(n_components, n_features);
        sigma.resize(n_components);

        // Linspace initialization for means
        for (int k = 0; k < n_components; ++k) {
            int idx = (n_samples - 1) * k / std::max(1, n_components - 1);
            mu.row(k) = X.row(idx);
        }

        // Global variance for initial sigma
        Eigen::RowVectorXd mean = X.colwise().mean();
        Eigen::RowVectorXd var = ((X.rowwise() - mean).array().square().colwise().sum() / n_samples) + 1e-6;
        
        for (int k = 0; k < n_components; ++k) {
            sigma[k] = var.transpose();
        }
    }

    Eigen::VectorXd gaussian_log_pdf(const Eigen::MatrixXd& X, const Eigen::VectorXd& mu_k, const Eigen::VectorXd& var_k) const {
        int n_samples = X.rows();
        int n_features = X.cols();
        Eigen::VectorXd log_pdf(n_samples);
        
        double log_det = var_k.array().log().sum();
        double const_term = -0.5 * n_features * std::log(2 * M_PI) - 0.5 * log_det;

        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            double exponent = -0.5 * ((X.row(i).transpose() - mu_k).array().square() / var_k.array()).sum();
            log_pdf(i) = const_term + exponent;
        }
        return log_pdf;
    }

    Eigen::MatrixXd e_step(const Eigen::MatrixXd& X) const {
        int n_samples = X.rows();
        Eigen::MatrixXd log_weighted_pdfs(n_samples, n_components);

        for (int k = 0; k < n_components; ++k) {
            log_weighted_pdfs.col(k) = gaussian_log_pdf(X, mu.row(k).transpose(), sigma[k]).array() + std::log(pi(k) + 1e-10);
        }

        Eigen::MatrixXd responsibilities(n_samples, n_components);
        
        // Log-sum-exp trick
        #pragma omp parallel for
        for (int i = 0; i < n_samples; ++i) {
            double max_val = log_weighted_pdfs.row(i).maxCoeff();
            Eigen::RowVectorXd exp_vals = (log_weighted_pdfs.row(i).array() - max_val).exp();
            responsibilities.row(i) = exp_vals / (exp_vals.sum() + 1e-10);
        }
        return responsibilities;
    }

    void m_step(const Eigen::MatrixXd& X, const Eigen::MatrixXd& resp) {
        int n_samples = X.rows();
        Eigen::RowVectorXd nk = resp.colwise().sum();
        pi = (nk / n_samples).transpose();

        #pragma omp parallel for
        for (int k = 0; k < n_components; ++k) {
            Eigen::VectorXd resp_k = resp.col(k);
            
            // Update mean
            mu.row(k) = (X.array().colwise() * resp_k.array()).colwise().sum() / nk(k);
            
            // Update variance (Diagonal)
            Eigen::MatrixXd diff = X.rowwise() - mu.row(k);
            Eigen::RowVectorXd var_k = (diff.array().square().colwise() * resp_k.array()).colwise().sum() / nk(k);
            sigma[k] = var_k.transpose().array() + 1e-6;
        }
    }

    double compute_log_likelihood(const Eigen::MatrixXd& X) const {
        int n_samples = X.rows();
        Eigen::MatrixXd log_weighted_pdfs(n_samples, n_components);

        for (int k = 0; k < n_components; ++k) {
            log_weighted_pdfs.col(k) = gaussian_log_pdf(X, mu.row(k).transpose(), sigma[k]).array() + std::log(pi(k) + 1e-10);
        }

        double total_ll = 0.0;
        
        #pragma omp parallel for reduction(+:total_ll)
        for (int i = 0; i < n_samples; ++i) {
            double max_val = log_weighted_pdfs.row(i).maxCoeff();
            double sum_exp = (log_weighted_pdfs.row(i).array() - max_val).exp().sum();
            total_ll += max_val + std::log(sum_exp + 1e-10);
        }
        return total_ll;
    }
};

#endif // GMM_HPP