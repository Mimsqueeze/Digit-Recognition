#ifndef DIGIT_RECOGNITION_NETWORK1_H
#define DIGIT_RECOGNITION_NETWORK1_H

#define WEIGHTS_AND_BIASES_FILE_PATH R"(.\src\double hidden layer\wandb.bin)"
#define L1_SIZE 300
#define L2_SIZE 100

#include <Eigen/Dense>

typedef struct {
    Eigen::MatrixXd W1, B1, W2, B2, W3, B3;
} weights_and_biases;

typedef struct {
    Eigen::MatrixXd Z1, A1, Z2, A2, Z3, A3;
} states_and_activations;

typedef struct {
    Eigen::MatrixXd dZ3, dW3, dB3, dZ2, dW2, dB2, dZ1, dW1, dB1;
} derivatives;

states_and_activations forward_prop(const Eigen::MatrixXd &X, const weights_and_biases &wab);

derivatives back_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const states_and_activations &saa,
                      const weights_and_biases &wab);

void update_params(weights_and_biases *wab, const derivatives &d, double learning_rate);

int gradient_descent(weights_and_biases &wab, double learning_rate, int epoch);

#endif //DIGIT_RECOGNITION_NETWORK1_H
