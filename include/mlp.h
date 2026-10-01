#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>

class MLP {
public:
    MLP(const std::vector<int>& sizes,
        const std::vector<double>& dropouts,
        unsigned seed = 42);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, bool training);
    void backward(const Eigen::MatrixXd& X, const Eigen::VectorXi& y,
                  const std::vector<double>& class_weights);
    void adam_step(double lr);

    Eigen::VectorXi predict(const Eigen::MatrixXd& X);
    double loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);

private:
    std::vector<Eigen::MatrixXd> W, dW, mW, vW;
    std::vector<Eigen::VectorXd> b, db, mb, vb;
    std::vector<double> dropout;

    std::vector<Eigen::MatrixXd> A;
    std::vector<Eigen::MatrixXd> Z;
    std::vector<Eigen::MatrixXd> M;

    int t = 0;
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    std::mt19937 rng;

    static Eigen::MatrixXd relu(const Eigen::MatrixXd& x) { return x.cwiseMax(0.0); }
    static Eigen::MatrixXd softmax(const Eigen::MatrixXd& x) {
        Eigen::MatrixXd out(x.rows(), x.cols());
        for (int i = 0; i < x.rows(); ++i) {
            double mx = x.row(i).maxCoeff();
            Eigen::VectorXd e = (x.row(i).array() - mx).exp();
            double s = e.sum();
            out.row(i) = (e / s).transpose();
        }
        return out;
    }
};
