#include <iostream>
#include "network1.h"
#include "../functions.h"
#include <Eigen/Dense>
#include <random>
#include <chrono>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

fp_return forward_prop(const MatrixXd &X, const MatrixXd &W1, const MatrixXd &B1, const MatrixXd &W2, const MatrixXd &B2) {
    fp_return fp;
    fp.Z1 = W1 * X + B1 * MatrixXd::Ones(1, BATCH_SIZE);
    if (ACTIVATION_FUNCTION == TANH)
        fp.A1 = fp.Z1.array().tanh();
    else if (ACTIVATION_FUNCTION == RELU)
        fp.A1 = fp.Z1.array().unaryExpr(&ReLU);
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        fp.A1 = fp.Z1.array().unaryExpr(&leaky_ReLU);
    fp.Z2 = W2 * fp.A1 + B2 * MatrixXd::Ones(1, BATCH_SIZE);
    fp.A2 = softmax(fp.Z2);

    return fp;
}

bp_return back_prop(const MatrixXd &X, const MatrixXd &Y, const MatrixXd &Z1, const MatrixXd &A1, const MatrixXd &A2,
                    const MatrixXd &W2) {
    bp_return bp;
    bp.dZ2 = A2 - Y;
    bp.dW2 = (1.0 / BATCH_SIZE) * bp.dZ2 * A1.transpose();
    bp.dB2 = (1.0 / BATCH_SIZE) * bp.dZ2.rowwise().sum();
    if (ACTIVATION_FUNCTION == TANH)
        bp.dZ1 = (W2.transpose() * bp.dZ2).cwiseProduct(deriv_tanh(Z1));
    else if (ACTIVATION_FUNCTION == RELU)
        bp.dZ1 = (W2.transpose() * bp.dZ2).cwiseProduct((MatrixXd) Z1.array().unaryExpr(&deriv_ReLU));
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        bp.dZ1 = (W2.transpose() * bp.dZ2).cwiseProduct((MatrixXd) Z1.array().unaryExpr(&deriv_leaky_ReLU));
    bp.dW1 = (1.0 / BATCH_SIZE) * bp.dZ1 * X.transpose();
    bp.dB1 = (1.0 / BATCH_SIZE) * bp.dZ1.rowwise().sum();
    return bp;
}

void update_params(MatrixXd *W1, MatrixXd *B1, MatrixXd *W2, MatrixXd *B2, const MatrixXd &dW1, const MatrixXd &dB1,
                   const MatrixXd &dW2, const MatrixXd &dB2, double learning_rate) {
    *W1 = *W1 - learning_rate * dW1;
    *B1 = *B1 - learning_rate * dB1;
    *W2 = *W2 - learning_rate * dW2;
    *B2 = *B2 - learning_rate * dB2;
}

void gradient_descent(MatrixXd *W1, MatrixXd *B1, MatrixXd *W2, MatrixXd *B2, double learning_rate, int epoch) {
    MatrixXd dW1 = MatrixXd::Zero(L1_SIZE, 784);
    MatrixXd dB1 = MatrixXd::Zero(L1_SIZE, 1);
    MatrixXd dW2 = MatrixXd::Zero(10, L1_SIZE);
    MatrixXd dB2 = MatrixXd::Zero(10, 1);
    int count = 0;

    int data_offsets[NUM_TRAIN_IMAGES];
    iota(data_offsets, data_offsets + NUM_TRAIN_IMAGES, 0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(data_offsets, data_offsets + NUM_BATCHES, std::default_random_engine(seed));

    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) {
        MatrixXd X = get_image_batch(data_offsets, i, BATCH_SIZE, TRAIN_IMAGES_FILE_PATH);
        MatrixXd Y = get_label_batch(data_offsets, i, BATCH_SIZE, TRAIN_LABELS_FILE_PATH);
        if (PRINT_IMAGE_AND_LABEL)
            print_batch(X, Y, BATCH_SIZE);
        fp_return fp = forward_prop(X, *W1, *B1, *W2, *B2);
        bp_return bp = back_prop(X, Y, fp.Z1, fp.A1, fp.A2, *W2);
        dW1 += bp.dW1;
        dB1 += bp.dB1;
        dW2 += bp.dW2;
        dB2 += bp.dB2;
        count += get_num_correct(get_predictions(fp.A2, BATCH_SIZE), Y, BATCH_SIZE);
    }

    dW1 /= NUM_BATCHES;
    dB1 /= NUM_BATCHES;
    dW2 /= NUM_BATCHES;
    dB2 /= NUM_BATCHES;

    update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate);
    cout << "Epoch: " << epoch << "\n";
    cout << "Accuracy: " << count << "/" << NUM_TRAIN_IMAGES << "\n";
}