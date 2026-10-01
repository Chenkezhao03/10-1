cmake_minimum_required(VERSION 3.15)
project(coastal_classifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} include)

add_executable(train_mlp
    src/main_mlp.cpp src/data_loader.cpp src/metrics.cpp src/mlp.cpp)

add_executable(train_cnn
    src/main_cnn.cpp src/data_loader.cpp src/metrics.cpp src/cnn1d.cpp)
