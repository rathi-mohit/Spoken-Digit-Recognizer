#ifndef HMM_HPP
#define HMM_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>

class HiddenMarkovModel {
public:
    int n_states;
    int n_observations;
    int max_iter;
    double tol;
    int random_state;

    Eigen::VectorXd pi;
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    std::vector<double> log_likelihood_history;

    HiddenMarkovModel(int n_states = 5, int n_observations = 32, int max_iter = 100, double tol = 1e-6, int random_state = 42)
        : n_states(n_states), n_observations(n_observations), max_iter(max_iter), tol(tol), random_state(random_state) {}

    void fit(const std::vector<std::vector<int>>& observations_list) {
        initialize_parameters();

        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd pi_acc = Eigen::VectorXd::Zero(n_states);
            Eigen::MatrixXd A_num = Eigen::MatrixXd::Zero(n_states, n_states);
            Eigen::VectorXd A_den = Eigen::VectorXd::Zero(n_states);
            Eigen::MatrixXd B_num = Eigen::MatrixXd::Zero(n_states, n_observations);
            Eigen::VectorXd B_den = Eigen::VectorXd::Zero(n_states);

            double total_ll = 0.0;

            for (const auto& obs : observations_list) {
                if (obs.empty()) continue;

                Eigen::MatrixXd alpha;
                Eigen::VectorXd scale;
                double log_likelihood;

                forward(obs, alpha, scale, log_likelihood);
                Eigen::MatrixXd beta = backward(obs, scale);
                total_ll += log_likelihood;

                // Gamma calculation
                Eigen::MatrixXd gamma = alpha.array() * beta.array();
                Eigen::VectorXd gamma_sum = gamma.rowwise().sum().array() + 1e-10;
                for (int t = 0; t < gamma.rows(); ++t) {
                    gamma.row(t) /= gamma_sum(t);
                }

                // Accumulate Pi
                pi_acc += gamma.row(0).transpose();

                // Accumulate A and B
                int T = obs.size();
                for (int t = 0; t < T - 1; ++t) {
                    Eigen::MatrixXd xi = (alpha.row(t).transpose() * (B.col(obs[t + 1]).transpose().array() * beta.row(t + 1).array()).matrix()).array() * A.array();
                    xi /= (xi.sum() + 1e-10);
                    A_num += xi;
                }
                
                A_den += gamma.topRows(T - 1).colwise().sum().transpose();
                B_den += gamma.colwise().sum().transpose();

                for (int t = 0; t < T; ++t) {
                    B_num.col(obs[t]) += gamma.row(t).transpose();
                }
            }

            // M-Step Updates
            pi = pi_acc.array() / (pi_acc.sum() + 1e-10);
            
            for (int i = 0; i < n_states; ++i) {
                A.row(i) = A_num.row(i).array() / (A_den(i) + 1e-10);
                B.row(i) = B_num.row(i).array() / (B_den(i) + 1e-10);
            }

            log_likelihood_history.push_back(total_ll);
        }
    }

    double score(const std::vector<int>& obs) const {
        Eigen::MatrixXd alpha;
        Eigen::VectorXd scale;
        double log_likelihood;
        forward(obs, alpha, scale, log_likelihood);
        return log_likelihood;
    }

private:
    void initialize_parameters() {
        std::mt19937 gen(random_state);
        std::uniform_real_distribution<> dis(0.0, 1.0);

        pi = Eigen::VectorXd(n_states);
        A = Eigen::MatrixXd(n_states, n_states);
        B = Eigen::MatrixXd(n_states, n_observations);

        for (int i = 0; i < n_states; ++i) {
            pi(i) = dis(gen);
            for (int j = 0; j < n_states; ++j) A(i, j) = dis(gen);
            for (int k = 0; k < n_observations; ++k) B(i, k) = dis(gen);
        }

        pi /= pi.sum();
        for (int i = 0; i < n_states; ++i) {
            A.row(i) /= A.row(i).sum();
            B.row(i) /= B.row(i).sum();
        }
    }

    void forward(const std::vector<int>& obs, Eigen::MatrixXd& alpha, Eigen::VectorXd& scale, double& log_likelihood) const {
        int T = obs.size();
        alpha = Eigen::MatrixXd::Zero(T, n_states);
        scale = Eigen::VectorXd::Zero(T);

        alpha.row(0) = (pi.array() * B.col(obs[0]).array()).transpose();
        scale(0) = 1.0 / (alpha.row(0).sum() + 1e-10);
        alpha.row(0) *= scale(0);

        for (int t = 1; t < T; ++t) {
            alpha.row(t) = (alpha.row(t - 1) * A).array() * B.col(obs[t]).transpose().array();
            scale(t) = 1.0 / (alpha.row(t).sum() + 1e-10);
            alpha.row(t) *= scale(t);
        }

        log_likelihood = -(scale.array() + 1e-10).log().sum();
    }

    Eigen::MatrixXd backward(const std::vector<int>& obs, const Eigen::VectorXd& scale) const {
        int T = obs.size();
        Eigen::MatrixXd beta = Eigen::MatrixXd::Zero(T, n_states);

        beta.row(T - 1) = Eigen::RowVectorXd::Constant(n_states, scale(T - 1));

        for (int t = T - 2; t >= 0; --t) {
            beta.row(t) = (A * (B.col(obs[t + 1]).array() * beta.row(t + 1).transpose().array()).matrix()).transpose() * scale(t);
        }

        return beta;
    }
};

#endif // HMM_HPP