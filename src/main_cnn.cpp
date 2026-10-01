#include "data_loader.h"
#include "cnn1d.h"
#include "metrics.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>

int main() {
    auto tr = load_csv("data/processed/train.csv");
    auto va = load_csv("data/processed/val.csv");
    auto te = load_csv("data/processed/test.csv");

    std::vector<std::string> class_names;
    { std::ifstream f("data/processed/classes.txt"); std::string s;
      while (std::getline(f, s)) if (!s.empty()) class_names.push_back(s); }

    std::cout << "Features: " << tr.n_features
              << "  Classes: " << tr.n_classes << "\n";

    Eigen::VectorXi cnt = Eigen::VectorXi::Zero(tr.n_classes);
    for (int i = 0; i < tr.n_samples; ++i) cnt(tr.y(i))++;
    std::vector<double> class_w(tr.n_classes);
    for (int k = 0; k < tr.n_classes; ++k)
        class_w[k] = (double)tr.n_samples / (tr.n_classes * std::max(1, cnt(k)));
    std::cout << "Class weights: ";
    for (auto w : class_w) std::cout << w << " ";
    std::cout << "\n";

    CNN1D model(tr.n_features, tr.n_classes);

    int EPOCHS = 30, BATCH = 32;

    auto t0 = std::chrono::steady_clock::now();

    for (int ep = 0; ep < EPOCHS; ++ep) {
        shuffle_dataset(tr, 42 + ep);
        for (int s = 0; s < tr.n_samples; s += BATCH) {
            int e = std::min(s + BATCH, tr.n_samples);
            Eigen::MatrixXd Xb = tr.X.middleRows(s, e - s);
            Eigen::VectorXi yb = tr.y.segment(s, e - s);
            model.forward(Xb, true);
            model.backward(Xb, yb, class_w);
        }
        if (ep % 2 == 0 || ep == EPOCHS - 1) {
            double vloss = model.loss(va.X, va.y);
            auto vpred = model.predict(va.X);
            std::cout << "Epoch " << ep
                      << "  val_loss=" << vloss
                      << "  val_acc=" << accuracy(va.y, vpred) << "\n";
        }
    }
    double train_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    auto t1 = std::chrono::steady_clock::now();
    auto pred = model.predict(te.X);
    double infer_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count() / te.n_samples;

    std::cout << "\n=== CNN TEST ===\n";
    std::cout << "Accuracy:       " << accuracy(te.y, pred) << "\n";
    std::cout << "Training time:  " << train_time << " s\n";
    std::cout << "Inference:      " << infer_ms << " ms/sample\n";
    print_report(te.y, pred, tr.n_classes, class_names);
    return 0;
}
