#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>

class CNN1D {
public:
    CNN1D(int D, int K, unsigned seed = 42);

    Eigen::MatrixXd forward(const Eigen::MatrixXd& X, bool training);
    void backward(const Eigen::MatrixXd& X, const Eigen::VectorXi& y,
                  const std::vector<double>& class_weights);

    Eigen::VectorXi predict(const Eigen::MatrixXd& X);
    double loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);

private:
    int D, K;
    int F1 = 32, F2 = 64, KSZ = 3, PAD = 1;
    int L1, L1p, L2, L2p, FLAT;

    std::vector<Eigen::MatrixXd> Wc1, mWc1, vWc1;
    Eigen::VectorXd bc1, mbc1, vbc1;

    std::vector<Eigen::MatrixXd> Wc2, mWc2, vWc2;
    Eigen::VectorXd bc2, mbc2, vbc2;

    Eigen::MatrixXd W1, mW1, vW1;
    Eigen::VectorXd b1, mb1, vb1;
    Eigen::MatrixXd W2, mW2, vW2;
    Eigen::VectorXd b2, mb2, vb2;

    int t = 0;
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    std::mt19937 rng;

    std::vector<Eigen::MatrixXd> Zc1, Ac1, Ap1, M1;
    std::vector<Eigen::MatrixXd> Zc2, Ac2, Ap2, M2;
    Eigen::MatrixXd flat, h1, mask_dense;

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
