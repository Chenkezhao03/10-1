#pragma once
#include <Eigen/Dense>
#include <string>

struct Dataset {
    Eigen::MatrixXd X;
    Eigen::VectorXi y;
    int n_samples = 0, n_features = 0, n_classes = 0;
};

Dataset load_csv(const std::string& path);
void shuffle_dataset(Dataset& ds, unsigned seed = 42);
