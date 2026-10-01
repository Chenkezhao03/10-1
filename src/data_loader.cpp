#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <random>

Dataset load_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string v;
        std::vector<double> row;
        while (std::getline(ss, v, ',')) row.push_back(std::stod(v));
        rows.push_back(row);
    }

    Dataset ds;
    ds.n_samples = rows.size();
    ds.n_features = rows[0].size() - 1;
    ds.X.resize(ds.n_samples, ds.n_features);
    ds.y.resize(ds.n_samples);

    int maxl = 0;
    for (int i = 0; i < ds.n_samples; ++i) {
        for (int j = 0; j < ds.n_features; ++j) ds.X(i, j) = rows[i][j];
        ds.y(i) = (int)rows[i][ds.n_features];
        maxl = std::max(maxl, ds.y(i));
    }
    ds.n_classes = maxl + 1;
    return ds;
}

void shuffle_dataset(Dataset& ds, unsigned seed) {
    std::vector<int> idx(ds.n_samples);
    for (int i = 0; i < ds.n_samples; ++i) idx[i] = i;
    std::mt19937 g(seed);
    std::shuffle(idx.begin(), idx.end(), g);
    Eigen::MatrixXd X2(ds.n_samples, ds.n_features);
    Eigen::VectorXi y2(ds.n_samples);
    for (int i = 0; i < ds.n_samples; ++i) {
        X2.row(i) = ds.X.row(idx[i]);
        y2(i) = ds.y(idx[i]);
    }
    ds.X = X2; ds.y = y2;
}
