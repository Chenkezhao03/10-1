#include "cnn1d.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

CNN1D::CNN1D(int D_, int K_, unsigned seed) : D(D_), K(K_), rng(seed) {
    L1  = D + 2 * PAD - KSZ + 1;
    L1p = L1 / 2;
    L2  = L1p + 2 * PAD - KSZ + 1;
    L2p = L2 / 2;
    FLAT = F2 * L2p;

    double s1 = std::sqrt(2.0 / KSZ);
    double s2 = std::sqrt(2.0 / (F1 * KSZ));
    double s3 = std::sqrt(2.0 / FLAT);
    double s4 = std::sqrt(2.0 / 128);
    std::normal_distribution<double> d1(0, s1), d2(0, s2), d3(0, s3), d4(0, s4);

    Wc1.resize(F1); mWc1.resize(F1); vWc1.resize(F1);
    for (int f = 0; f < F1; ++f) {
        Wc1[f] = Eigen::MatrixXd::NullaryExpr(KSZ, 1, [&](){ return d1(rng); });
        mWc1[f] = Eigen::MatrixXd::Zero(KSZ, 1);
        vWc1[f] = Eigen::MatrixXd::Zero(KSZ, 1);
    }
    bc1 = Eigen::VectorXd::Zero(F1);
    mbc1 = Eigen::VectorXd::Zero(F1); vbc1 = Eigen::VectorXd::Zero(F1);

    Wc2.resize(F2); mWc2.resize(F2); vWc2.resize(F2);
    for (int f = 0; f < F2; ++f) {
        Wc2[f] = Eigen::MatrixXd::NullaryExpr(F1, KSZ, [&](){ return d2(rng); });
        mWc2[f] = Eigen::MatrixXd::Zero(F1, KSZ);
        vWc2[f] = Eigen::MatrixXd::Zero(F1, KSZ);
    }
    bc2 = Eigen::VectorXd::Zero(F2);
    mbc2 = Eigen::VectorXd::Zero(F2); vbc2 = Eigen::VectorXd::Zero(F2);

    W1 = Eigen::MatrixXd::NullaryExpr(FLAT, 128, [&](){ return d3(rng); });
    b1 = Eigen::VectorXd::Zero(128);
    W2 = Eigen::MatrixXd::NullaryExpr(128, K, [&](){ return d4(rng); });
    b2 = Eigen::VectorXd::Zero(K);

    mW1 = Eigen::MatrixXd::Zero(FLAT, 128); vW1 = Eigen::MatrixXd::Zero(FLAT, 128);
    mb1 = Eigen::VectorXd::Zero(128); vb1 = Eigen::VectorXd::Zero(128);
    mW2 = Eigen::MatrixXd::Zero(128, K); vW2 = Eigen::MatrixXd::Zero(128, K);
    mb2 = Eigen::VectorXd::Zero(K); vb2 = Eigen::VectorXd::Zero(K);
}

static Eigen::MatrixXd conv1d_sample(const Eigen::MatrixXd& xin,
                                     const std::vector<Eigen::MatrixXd>& W,
                                     const Eigen::VectorXd& b,
                                     int F, int KSZ, int PAD, int Lout)
{
    int Cin = xin.rows();
    int Lin = xin.cols();
    Eigen::MatrixXd out(F, Lout);
    for (int o = 0; o < F; ++o) {
        for (int i = 0; i < Lout; ++i) {
            double s = b(o);
            for (int c = 0; c < Cin; ++c)
                for (int k = 0; k < KSZ; ++k) {
                    int xi = i + k - PAD;
                    if (xi >= 0 && xi < Lin) s += xin(c, xi) * W[o](c, k);
                }
            out(o, i) = s;
        }
    }
    return out;
}

Eigen::MatrixXd CNN1D::forward(const Eigen::MatrixXd& X, bool training) {
    int N = X.rows();
    Zc1.clear(); Ac1.clear(); Ap1.clear(); M1.clear();
    Zc2.clear(); Ac2.clear(); Ap2.clear(); M2.clear();

    std::uniform_real_distribution<double> u01(0, 1);
    auto make_mask = [&](int r, int c, double p) {
        return Eigen::MatrixXd::NullaryExpr(r, c,
            [&](){ return u01(rng) > p ? 1.0/(1.0-p) : 0.0; });
    };

    std::vector<Eigen::MatrixXd> pooled1;
    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd xcol = X.row(n).transpose();
        Eigen::MatrixXd z1 = conv1d_sample(xcol, Wc1, bc1, F1, KSZ, PAD, L1);
        Eigen::MatrixXd a1 = relu(z1);
        Eigen::MatrixXd p1(F1, L1p);
        for (int f = 0; f < F1; ++f)
            for (int i = 0; i < L1p; ++i)
                p1(f, i) = std::max(a1(f, 2*i), a1(f, 2*i+1));
        Eigen::MatrixXd m1;
        if (training) { m1 = make_mask(F1, L1p, 0.2); p1 = p1.array() * m1.array(); }
        Zc1.push_back(z1); Ac1.push_back(a1); Ap1.push_back(p1); M1.push_back(m1);
        pooled1.push_back(p1);
    }

    std::vector<Eigen::MatrixXd> pooled2;
    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd z2 = conv1d_sample(pooled1[n], Wc2, bc2, F2, KSZ, PAD, L2);
        Eigen::MatrixXd a2 = relu(z2);
        Eigen::MatrixXd p2(F2, L2p);
        for (int f = 0; f < F2; ++f)
            for (int i = 0; i < L2p; ++i)
                p2(f, i) = std::max(a2(f, 2*i), a2(f, 2*i+1));
        Eigen::MatrixXd m2;
        if (training) { m2 = make_mask(F2, L2p, 0.2); p2 = p2.array() * m2.array(); }
        Zc2.push_back(z2); Ac2.push_back(a2); Ap2.push_back(p2); M2.push_back(m2);
        pooled2.push_back(p2);
    }

    flat.resize(N, FLAT);
    for (int n = 0; n < N; ++n)
        for (int f = 0; f < F2; ++f)
            for (int i = 0; i < L2p; ++i)
                flat(n, f * L2p + i) = pooled2[n](f, i);

    h1 = (flat * W1).rowwise() + b1.transpose();
    h1 = relu(h1);
    if (training) {
        mask_dense = make_mask(N, 128, 0.3);
        h1 = h1.array() * mask_dense.array();
    } else {
        mask_dense = Eigen::MatrixXd();
    }

    Eigen::MatrixXd out = (h1 * W2).rowwise() + b2.transpose();
    return softmax(out);
}

void CNN1D::backward(const Eigen::MatrixXd& X, const Eigen::VectorXi& y,
                     const std::vector<double>& class_weights)
{
    int N = X.rows();

    Eigen::MatrixXd p = (h1 * W2).rowwise() + b2.transpose();
    p = softmax(p);
    Eigen::MatrixXd dOut = p;
    for (int i = 0; i < N; ++i) {
        dOut(i, y(i)) -= 1.0;
        double w = (y(i) < (int)class_weights.size()) ? class_weights[y(i)] : 1.0;
        dOut.row(i) *= w;
    }
    dOut /= (double)N;

    Eigen::MatrixXd dW2 = h1.transpose() * dOut;
    Eigen::VectorXd db2 = dOut.colwise().sum();

    Eigen::MatrixXd dh1 = dOut * W2.transpose();
    if (mask_dense.size() > 0) dh1 = dh1.array() * mask_dense.array();
    Eigen::MatrixXd dRelu = dh1.array() * (h1.array() > 0).cast<double>();
    Eigen::MatrixXd dW1 = flat.transpose() * dRelu;
    Eigen::VectorXd db1 = dRelu.colwise().sum();
    Eigen::MatrixXd dflat = dRelu * W1.transpose();

    std::vector<Eigen::MatrixXd> dAp2(N);
    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd dA(F2, L2p);
        for (int f = 0; f < F2; ++f)
            for (int i = 0; i < L2p; ++i)
                dA(f, i) = dflat(n, f * L2p + i);
        if (M2[n].size() > 0) dA = dA.array() * M2[n].array();
        dAp2[n] = dA;
    }

    std::vector<Eigen::MatrixXd> dAc2(N);
    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd dA = Eigen::MatrixXd::Zero(F2, L2);
        for (int f = 0; f < F2; ++f)
            for (int i = 0; i < L2p; ++i) {
                if (Ac2[n](f, 2*i) >= Ac2[n](f, 2*i+1)) dA(f, 2*i)   = dAp2[n](f, i);
                else                                    dA(f, 2*i+1) = dAp2[n](f, i);
            }
        dAc2[n] = dA;
    }

    std::vector<Eigen::MatrixXd> dZ2(N);
    for (int n = 0; n < N; ++n)
        dZ2[n] = dAc2[n].array() * (Zc2[n].array() > 0).cast<double>();

    std::vector<Eigen::MatrixXd> dWc2_new(F2, Eigen::MatrixXd::Zero(F1, KSZ));
    Eigen::VectorXd dbc2_new = Eigen::VectorXd::Zero(F2);
    std::vector<Eigen::MatrixXd> dAp1(N, Eigen::MatrixXd::Zero(F1, L1p));

    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd xin = Ap1[n];
        for (int o = 0; o < F2; ++o)
            for (int i = 0; i < L2; ++i) {
                double g = dZ2[n](o, i);
                dbc2_new(o) += g;
                for (int c = 0; c < F1; ++c)
                    for (int k = 0; k < KSZ; ++k) {
                        int xi = i + k - PAD;
                        if (xi >= 0 && xi < L1p) {
                            dWc2_new[o](c, k) += g * xin(c, xi);
                            dAp1[n](c, xi) += g * Wc2[o](c, k);
                        }
                    }
            }
    }

    std::vector<Eigen::MatrixXd> dAc1(N);
    for (int n = 0; n < N; ++n) {
        Eigen::MatrixXd dA = Eigen::MatrixXd::Zero(F1, L1);
        for (int f = 0; f < F1; ++f)
            for (int i = 0; i < L1p; ++i) {
                if (Ac1[n](f, 2*i) >= Ac1[n](f, 2*i+1)) dA(f, 2*i)   = dAp1[n](f, i);
                else                                    dA(f, 2*i+1) = dAp1[n](f, i);
            }
        dAc1[n] = dA;
    }

    std::vector<Eigen::MatrixXd> dZ1(N);
    for (int n = 0; n < N; ++n)
        dZ1[n] = dAc1[n].array() * (Zc1[n].array() > 0).cast<double>();

    std::vector<Eigen::MatrixXd> dWc1_new(F1, Eigen::MatrixXd::Zero(KSZ, 1));
    Eigen::VectorXd dbc1_new = Eigen::VectorXd::Zero(F1);
    for (int n = 0; n < N; ++n) {
        Eigen::VectorXd xrow = X.row(n).transpose();
        for (int o = 0; o < F1; ++o)
            for (int i = 0; i < L1; ++i) {
                double g = dZ1[n](o, i);
                dbc1_new(o) += g;
                for (int k = 0; k < KSZ; ++k) {
                    int xi = i + k - PAD;
                    if (xi >= 0 && xi < D) dWc1_new[o](k, 0) += g * xrow(xi);
                }
            }
    }

    t++;
    double bc1_ = 1 - std::pow(beta1, t);
    double bc2_ = 1 - std::pow(beta2, t);
    double lr = 1e-3;

    auto upd_m = [&](Eigen::MatrixXd& W_, Eigen::MatrixXd& m, Eigen::MatrixXd& v,
                     const Eigen::MatrixXd& g) {
        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g.array().square().matrix();
        W_.array() -= lr * (m / bc1_).array()
                     / ((v / bc2_).array().sqrt() + eps);
    };
    auto upd_v = [&](Eigen::VectorXd& W_, Eigen::VectorXd& m, Eigen::VectorXd& v,
                     const Eigen::VectorXd& g) {
        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g.array().square().matrix();
        W_.array() -= lr * (m / bc1_).array()
                     / ((v / bc2_).array().sqrt() + eps);
    };

    for (int f = 0; f < F1; ++f) upd_m(Wc1[f], mWc1[f], vWc1[f], dWc1_new[f]);
    upd_v(bc1, mbc1, vbc1, dbc1_new);
    for (int f = 0; f < F2; ++f) upd_m(Wc2[f], mWc2[f], vWc2[f], dWc2_new[f]);
    upd_v(bc2, mbc2, vbc2, dbc2_new);
    upd_m(W1, mW1, vW1, dW1);
    upd_v(b1, mb1, vb1, db1);
    upd_m(W2, mW2, vW2, dW2);
    upd_v(b2, mb2, vb2, db2);
}

Eigen::VectorXi CNN1D::predict(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd p = forward(X, false);
    Eigen::VectorXi out(p.rows());
    for (int i = 0; i < p.rows(); ++i) {
        int idx; p.row(i).maxCoeff(&idx);
        out(i) = idx;
    }
    return out;
}

double CNN1D::loss(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    Eigen::MatrixXd p = forward(X, false);
    double L = 0;
    for (int i = 0; i < p.rows(); ++i) L -= std::log(p(i, y(i)) + 1e-9);
    return L / p.rows();
}
