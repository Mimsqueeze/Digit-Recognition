#ifndef DIGIT_RECOGNITION_NETWORK1_H
#define DIGIT_RECOGNITION_NETWORK1_H

#define WEIGHTS_AND_BIASES_FILE_PATH R"(.\src\single hidden layer\wandb.bin)"
#define L1_SIZE 250

#include <Eigen/Dense>

typedef struct {
    Eigen::MatrixXd Z1, A1, Z2, A2;
} states_and_activations;

typedef struct {
    Eigen::MatrixXd dZ2, dW2, dB2, dZ1, dW1, dB1;
} derivatives;

states_and_activations forward_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &W1, const Eigen::MatrixXd &B1,
                                    const Eigen::MatrixXd &W2, const Eigen::MatrixXd &B2);

derivatives back_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::MatrixXd &Z1,
                                   const Eigen::MatrixXd &A1, const Eigen::MatrixXd &Z2, const Eigen::MatrixXd &A2,
                                   const Eigen::MatrixXd &W2);

void update_params(Eigen::MatrixXd *W1, Eigen::MatrixXd *B1, Eigen::MatrixXd *W2, Eigen::MatrixXd *B2,
                   const Eigen::MatrixXd &dW1, const Eigen::MatrixXd &dB1, const Eigen::MatrixXd &dW2,
                   const Eigen::MatrixXd &dB2, double learning_rate);

int gradient_descent(Eigen::MatrixXd *W1, Eigen::MatrixXd *B1, Eigen::MatrixXd *W2, Eigen::MatrixXd *B2,
                      double learning_rate, int epoch);

#endif //DIGIT_RECOGNITION_NETWORK1_H
