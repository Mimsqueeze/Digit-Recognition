#include <iostream>
#include "network2.h"
#include "../functions.h"
#include <Eigen/Dense>
#include <random>
#include <chrono>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

states_and_activations forward_prop(const Eigen::MatrixXd &X, const weights_and_biases &wab) {
    // Initialize return struct
    states_and_activations fp;

    // Neuron state:
    // For all neurons in layer l, its state is defined as its bias plus all incoming connections
    // (weight * activation from all source neurons)
    // Zl = Wl * Al-1 + Bl

    // Neuron activation:
    // For all neurons in layer l, its activation is defined as its state passed through an activation function f
    // Al = f(Zl)

    // Layer 1 (Hidden layer 1)
    fp.Z1 = wab.W1 * X + wab.B1 * MatrixXd::Ones(1, BATCH_SIZE);
    if (ACTIVATION_FUNCTION == TANH)
        fp.A1 = fp.Z1.array().tanh();
    else if (ACTIVATION_FUNCTION == RELU)
        fp.A1 = fp.Z1.array().unaryExpr(&ReLU);
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        fp.A1 = fp.Z1.array().unaryExpr(&leaky_ReLU);

    // Layer 2 (Hidden layer 2)
    fp.Z2 = wab.W2 * fp.A1 + wab.B2 * MatrixXd::Ones(1, BATCH_SIZE);
    if (ACTIVATION_FUNCTION == TANH)
        fp.A2 = fp.Z2.array().tanh();
    else if (ACTIVATION_FUNCTION == RELU)
        fp.A2 = fp.Z2.array().unaryExpr(&ReLU);
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        fp.A2 = fp.Z2.array().unaryExpr(&leaky_ReLU);


    // Layer 3 (Output layer)
    fp.Z3 = wab.W3 * fp.A2 + wab.B3 * MatrixXd::Ones(1, BATCH_SIZE);
    fp.A3 = softmax(fp.Z3);

    return fp;
}

derivatives back_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const states_and_activations &saa,
                      const weights_and_biases &wab) {
    // Initialize return struct
    derivatives bp;

    // Cost function:
    // C = 1/2(Y - AL)^2

    // Change of cost relative to change in state, where L is the last layer:
    // dC/dZL = dC/dAL             (*) f'(zL)
    // dC/dZl = (Wl+1)T * dC/dZl+1 (*) f'(zl)

    // Change of cost relative to change in weights
    // dC/dWl = Al-1 * dC/dZl

    // Change of cost relative to change in biases
    // dC/dBl = dC/dZl

    // Compute derivatives for Layer 3 (Output layer)
    bp.dZ3 = saa.A3 - Y;
    bp.dW3 = (1.0 / BATCH_SIZE) * bp.dZ3 * saa.A2.transpose();
    bp.dB3 = (1.0 / BATCH_SIZE) * bp.dZ3.rowwise().sum();

    // Computes derivatives for Layer 2 (Hidden layer 2)
    if (ACTIVATION_FUNCTION == TANH)
        bp.dZ2 = (wab.W3.transpose() * bp.dZ3).cwiseProduct(deriv_tanh(saa.Z2));
    else if (ACTIVATION_FUNCTION == RELU)
        bp.dZ2 = (wab.W3.transpose() * bp.dZ3).cwiseProduct((MatrixXd) saa.Z2.array().unaryExpr(&deriv_ReLU));
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        bp.dZ2 = (wab.W3.transpose() * bp.dZ3).cwiseProduct((MatrixXd) saa.Z2.array().unaryExpr(&deriv_leaky_ReLU));
    bp.dW2 = (1.0 / BATCH_SIZE) * bp.dZ2 * X.transpose();
    bp.dB2 = (1.0 / BATCH_SIZE) * bp.dZ2.rowwise().sum();

    // Computes derivatives for Layer 1 (Hidden layer 1)
    if (ACTIVATION_FUNCTION == TANH)
        bp.dZ1 = (wab.W2.transpose() * bp.dZ2).cwiseProduct(deriv_tanh(saa.Z1));
    else if (ACTIVATION_FUNCTION == RELU)
        bp.dZ1 = (wab.W2.transpose() * bp.dZ2).cwiseProduct((MatrixXd) saa.Z1.array().unaryExpr(&deriv_ReLU));
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        bp.dZ1 = (wab.W2.transpose() * bp.dZ2).cwiseProduct((MatrixXd) saa.Z1.array().unaryExpr(&deriv_leaky_ReLU));
    bp.dW1 = (1.0 / BATCH_SIZE) * bp.dZ1 * X.transpose();
    bp.dB1 = (1.0 / BATCH_SIZE) * bp.dZ1.rowwise().sum();

    return bp;
}

void update_params(weights_and_biases *wab, const derivatives &d, double learning_rate) {
    // Update the weights and biases by moving in the direction of the steepest descent
    wab->W1 -= learning_rate * d.dW1;
    wab->B1 -= learning_rate * d.dB1;
    wab->W2 -= learning_rate * d.dW2;
    wab->B2 -= learning_rate * d.dB2;
    wab->W3 -= learning_rate * d.dW3;
    wab->B3 -= learning_rate * d.dB3;
}

int gradient_descent(weights_and_biases &wab, double learning_rate, int epoch) {
    // Initialize derivative matrices to 0.
    // Note, these are basically storing the "nudges" that will be done to W1, B1, W2, and B2
    derivatives d;
    d.dW1 = MatrixXd::Zero(L1_SIZE, 784);
    d.dB1 = MatrixXd::Zero(L1_SIZE, 1);
    d.dW2 = MatrixXd::Zero(L2_SIZE, L1_SIZE)/2;
    d.dB2 = MatrixXd::Zero(L2_SIZE, 1)/2;
    d.dW3 = MatrixXd::Zero(10, L2_SIZE);
    d.dB3 = MatrixXd::Zero(10, 1);

    // Initialize count variable to store number of correct predictions
    int count = 0;

    // Create array of offsets each associated with a label/image pair
    int data_offsets[NUM_TRAIN_IMAGES];

    // Fill with numbers 0 to 1-NUM_TRAIN_IMAGES in increasing order
    iota(data_offsets, data_offsets + NUM_TRAIN_IMAGES, 0);

    // Randomly shuffle array of offsets, to randomize image selection in mini-batches
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(data_offsets, data_offsets + NUM_BATCHES, std::default_random_engine(seed));

    // For every training image, go through gradient descent in mini-batches
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) {
        // Get image and label batch
        MatrixXd X = get_image_batch(data_offsets, i, BATCH_SIZE, TRAIN_IMAGES_FILE_PATH);
        MatrixXd Y = get_label_batch(data_offsets, i, BATCH_SIZE, TRAIN_LABELS_FILE_PATH);

        // Optionally print out the training labels and images
        if (PRINT_LABELS_AND_IMAGES)
            print_batch(X, Y, BATCH_SIZE);

        // Forward propagate to get Z1, A1, Z2, and A2
        states_and_activations fp = forward_prop(X, wab);

        // Back propagate to get dW1, dB1, dW2, dB2
        derivatives bp = back_prop(X, Y, fp, wab);

        // Add derivatives from mini-batch, in other words add the "nudges"
        d.dW1 += bp.dW1;
        d.dB1 += bp.dB1;
        d.dW2 += bp.dW2;
        d.dB2 += bp.dB2;
        d.dW3 += bp.dW3;
        d.dB3 += bp.dB3;

        // Add the number of correct predictions from the mini-batch
        count += get_num_correct(get_predictions(fp.A3, BATCH_SIZE), Y, BATCH_SIZE);
    }

    // Divide each derivative or "nudge" value by the number of batches to find the average among all batches
    d.dW1 /= NUM_BATCHES;
    d.dB1 /= NUM_BATCHES;
    d.dW2 /= NUM_BATCHES;
    d.dB2 /= NUM_BATCHES;
    d.dW3 /= NUM_BATCHES;
    d.dB3 /= NUM_BATCHES;

    // Update the parameters W1, B1, W2, and B2 with the "nudges"
    update_params(&wab, d, learning_rate);

    return count;
}