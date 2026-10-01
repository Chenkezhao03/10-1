#include "mlp.h"
#include <cmath>
#include <stdexcept>

MLP::MLP(const std::vector<int>& sizes,
         const std::vector<double>& dropouts,
         unsigned seed) : dropout(dropouts), rng(seed)
{
    int L = sizes.size() - 1;
    if ((int)dropouts.size() != L - 1) throw std::runtime_error("dropout size mismatch");

    W.resize(L); dW.resize(L); mW.resize(L); vW.resize(L);
    b.resize(L); db.resize(L); mb.resize(L); vb.resize(L);

    for (int i = 0; i < L; ++i) {
        double scale = std::sqrt(2.0 / sizes[i]);
        std::normal_distribution<double> dist(0, scale);
        W[i].resize(sizes[i+1], sizes[i]);
        for (int r = 0; r < W[i].rows(); ++r)
            for (int c = 0; c < W[i].cols(); ++c)
                W[i](r, c) = dist(rng);
        b[i] = Eigen::VectorXd::Zero(sizes[i+1]);

        dW[i] = Eigen::MatrixXd::Zero(sizes[i+1], sizes[i]);
        db[i] = Eigen::VectorXd::Zero(sizes[i+1]);
        mW[i] = Eigen::MatrixXd::Zero(sizes[i+1], sizes[i]);
        vW[i] = Eigen::MatrixXd::Zero(sizes[i+1], sizes[i]);
        mb[i] = Eigen::VectorXd::Zero(sizes[i+1]);
        vb[i] = Eigen::VectorXd::Zero(sizes[i+1]);
    }
}

Eigen::MatrixXd MLP::forward(const Eigen::MatrixXd& X, bool training) {
    int L = W.size();
    A.clear(); Z.clear(); M.clear();
    A.push_back(X);

    std::uniform_real_distribution<double> u01(0, 1);

    for (int i = 0; i < L - 1; ++i) {
        Eigen::MatrixXd z = (A[i] * W[i].transpose()).rowwise() + b[i].transpose();
        Z.push_back(z);
        Eigen::MatrixXd h = relu(z);

        if (training && dropout[i] > 0) {
            Eigen::MatrixXd m = Eigen::MatrixXd::NullaryExpr(
                h.rows(), h.cols(),
                [&]() { return u01(rng) > dropout[i] ? 1.0/(1.0-dropout[i]) : 0.0; });
            h = h.array() * m.array();
            M.push_back(m);
        } else {
            M.push_back(Eigen::MatrixXd());
        }
        A.push_back(h);
    }

    Eigen::MatrixXd z = (A[L-1] * W[L-1].transpose()).rowwise() + b[L-1].transpose();
    Z.push_back(z);
    A.push_back(softmax(z));
    return A[L];
}

void MLP::backward(const Eigen::MatrixXd& X, const Eigen::VectorXi& y,
                   const std::vector<double>& class_weights)
{
    int L = W.size();
    int N = X.rows();

    Eigen::MatrixXd d = A[L];
    for (int i = 0; i < N; ++i) {
        d(i, y(i)) -= 1.0;
        double w = (y(i) < (int)class_weights.size()) ? class_weights[y(i)] : 1.0;
        d.row(i) *= w;
    }
    d /= (double)N;

    for (int i = L - 1; i >= 0; --i) {
        dW[i] = d.transpose() * A[i];
        db[i] = d.colwise().sum();

        if (i > 0) {
            Eigen::MatrixXd dA = d * W[i];
            if (M[i-1].size() > 0) dA = dA.array() * M[i-1].array();
            d = dA.array() * (Z[i-1].array() > 0).cast<double>();
        }
    }
}

void MLP::adam_step(double lr) {
    t++;
    double bc1 = 1 - std::pow(beta1, t);
    double bc2 = 1 - std::pow(beta2, t);
    for (size_t i = 0; i < W.size(); ++i) {
        mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i];
        vW[i] = beta2 * vW[i] + (1 - beta2) * dW[i].array().square().matrix();
        W[i].array() -= lr * (mW[i] / bc1).array()
                       / ((vW[i] / bc2).array().sqrt() + eps);

        mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
        vb[i] = beta2 * vb[i] + (1 - beta2) * db[i].array().square().matrix();
        b[i].array() -= lr * (mb[i] / bc1).array()
                       / ((vb[i] / bc2).array().sqrt() + eps);
    }
}

Eigen::VectorXi MLP::predict(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd p = forward(X, false);
    Eigen::VectorXi out(p.rows());
    for (int i = 0; i < p.rows(); ++i) {
        int idx; p.row(i).maxCoeff(&idx);
        out(i) = idx;
    }
    return out;
}

double MLP::loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    Eigen::MatrixXd p = forward(X, false);
    double L = 0;
    for (int i = 0; i < p.rows(); ++i) L -= std::log(p(i, y(i)) + 1e-9);
    return L / p.rows();
}
