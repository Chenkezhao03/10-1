#include "data_loader.h"
#include "mlp.h"
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

    MLP model({tr.n_features, 256, 128, 64, tr.n_classes},
              {0.3, 0.3, 0.2});
    int EPOCHS = 120, BATCH = 32;
    double LR = 1e-3;
    int patience_lr = 10, lr_wait = 0;
    double best_val = 1e9;

    auto t0 = std::chrono::steady_clock::now();

    for (int ep = 0; ep < EPOCHS; ++ep) {
        shuffle_dataset(tr, 42 + ep);
        for (int s = 0; s < tr.n_samples; s += BATCH) {
            int e = std::min(s + BATCH, tr.n_samples);
            Eigen::MatrixXd Xb = tr.X.middleRows(s, e - s);
            Eigen::VectorXi yb = tr.y.segment(s, e - s);
            model.forward(Xb, true);
            model.backward(Xb, yb, class_w);
            model.adam_step(LR);
        }
        double vloss = model.loss(va.X, va.y);
        auto vpred = model.predict(va.X);
        double vacc = accuracy(va.y, vpred);

        if (ep % 5 == 0 || ep == EPOCHS - 1)
            std::cout << "Epoch " << ep
                      << "  val_loss=" << vloss
                      << "  val_acc=" << vacc << "\n";

        if (vloss < best_val - 1e-6) { best_val = vloss; lr_wait = 0; }
        else if (++lr_wait >= patience_lr && LR > 1e-5) {
            LR *= 0.5;
            std::cout << "  [ReduceLROnPlateau] LR -> " << LR << "\n";
            lr_wait = 0;
        }
    }
    double train_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    auto t1 = std::chrono::steady_clock::now();
    auto pred = model.predict(te.X);
    double infer_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count() / te.n_samples;

    std::cout << "\n=== MLP TEST ===\n";
    std::cout << "Accuracy:       " << accuracy(te.y, pred) << "\n";
    std::cout << "Training time:  " << train_time << " s\n";
    std::cout << "Inference:      " << infer_ms << " ms/sample\n";
    print_report(te.y, pred, tr.n_classes, class_names);
    return 0;
}
