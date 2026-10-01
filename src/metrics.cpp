#include "metrics.h"
#include <iostream>
#include <iomanip>

double accuracy(const Eigen::VectorXi& yt, const Eigen::VectorXi& yp) {
    int c = 0;
    for (int i = 0; i < yt.size(); ++i) if (yt(i) == yp(i)) c++;
    return (double)c / yt.size();
}

void print_report(const Eigen::VectorXi& yt, const Eigen::VectorXi& yp,
                  int K, const std::vector<std::string>& names) {
    Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(K, K);
    for (int i = 0; i < yt.size(); ++i) cm(yt(i), yp(i))++;

    std::cout << "\nConfusion matrix (rows=true, cols=pred):\n" << cm << "\n\n";

    double macro_p = 0, macro_r = 0, macro_f = 0;
    for (int k = 0; k < K; ++k) {
        int tp = cm(k, k);
        int fp = cm.col(k).sum() - tp;
        int fn = cm.row(k).sum() - tp;
        double p = tp + fp > 0 ? (double)tp / (tp + fp) : 0.0;
        double r = tp + fn > 0 ? (double)tp / (tp + fn) : 0.0;
        double f = p + r > 0 ? 2 * p * r / (p + r) : 0.0;
        macro_p += p; macro_r += r; macro_f += f;
        std::string name = k < (int)names.size() ? names[k] : ("class_" + std::to_string(k));
        std::cout << std::fixed << std::setprecision(4)
                  << std::left << std::setw(14) << name
                  << " Precision=" << p << "  Recall=" << r << "  F1=" << f << "\n";
    }
    std::cout << "\nMacro: Precision=" << macro_p/K
              << "  Recall=" << macro_r/K
              << "  F1=" << macro_f/K << "\n";
}
