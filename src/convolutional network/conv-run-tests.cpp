#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "../functions.h"
#include "conv-network.h"

using namespace std;
using Eigen::MatrixXd;

int main() {
    // Obtain the testing set
    MatrixXd X = get_images(0, NUM_TEST_IMAGES, TEST_IMAGES_FILE_PATH);
    MatrixXd Y = get_labels(0, NUM_TEST_IMAGES, TEST_LABELS_FILE_PATH);

    // Extract the weights and biases from file
    MatrixXd W1(L1_SIZE, CONVOLUTION_OUTPUT_SIZE);
    MatrixXd B1(L1_SIZE, 1);
    MatrixXd W2(10, L1_SIZE);
    MatrixXd B2(10, 1);

    streamoff read_position = 0;
    read_position = read(&W1, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read_position = read(&B1, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read_position = read(&W2, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read(&B2, read_position, WEIGHTS_AND_BIASES_FILE_PATH);

    // Convolve and pool on X
    MatrixXd C = getConvolution(X);

    // Do forward propagation with the stored weights and biases
    states_and_activations fp = forward_prop(C, W1, B1, W2, B2);

    // Get the number of correct predictions
    int count = get_num_correct(get_predictions(fp.A2, NUM_TEST_IMAGES), Y, NUM_TEST_IMAGES);

    // Optionally print out the test labels and images
    if (PRINT_LABELS_AND_IMAGES)
        print_batch(X, Y, NUM_TEST_IMAGES);

    // Print the accuracy of the trained neural network
    cout << "Accuracy: " << count << "/" << NUM_TEST_IMAGES << "\n";

    return 0;
}