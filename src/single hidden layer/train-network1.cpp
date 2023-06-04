#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "../functions.h"
#include "network1.h"

using namespace std;
using Eigen::MatrixXd;

int main() {
    // Randomize the starting seed
    srand((unsigned int) time(nullptr));

    // Initialize weights and biases to a random value between -0.5 and 0.5
    MatrixXd W1 = MatrixXd::Random(L1_SIZE, 784)/2;
    MatrixXd B1 = MatrixXd::Random(L1_SIZE, 1)/2;
    MatrixXd W2 = MatrixXd::Random(10, L1_SIZE)/2;
    MatrixXd B2 = MatrixXd::Random(10, 1)/2;

    // For each epoch, perform gradient descent and update weights and biases
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        gradient_descent(&W1, &B1, &W2, &B2, LEARNING_RATE, epoch);
    }

    // Save weights and biases to file
    streamoff write_position = 0;
    write_position = save(W1, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    write_position = save(B1, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    write_position = save(W2, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    save(B2, write_position, WEIGHTS_AND_BIASES_FILE_PATH);

    return 0;
}