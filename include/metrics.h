#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

double accuracy(const Eigen::VectorXi& yt, const Eigen::VectorXi& yp);
void print_report(const Eigen::VectorXi& yt, const Eigen::VectorXi& yp,
                  int K, const std::vector<std::string>& names = {});
