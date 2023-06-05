#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "../functions.h"
#include "network2.h"

using namespace std;
using Eigen::MatrixXd;

int main() {
    // Obtain the testing set
    MatrixXd X = get_images(0, NUM_TEST_IMAGES, TEST_IMAGES_FILE_PATH);
    MatrixXd Y = get_labels(0, NUM_TEST_IMAGES, TEST_LABELS_FILE_PATH);

    // Extract the weights and biases from file
    weights_and_biases wab;
    wab.W1 = MatrixXd::Random(L1_SIZE, 784)/2;
    wab.B1 = MatrixXd::Random(L1_SIZE, 1)/2;
    wab.W2 = MatrixXd::Random(L2_SIZE, L1_SIZE)/2;
    wab.B2 = MatrixXd::Random(L2_SIZE, 1)/2;
    wab.W3 = MatrixXd::Random(10, L2_SIZE)/2;
    wab.B3 = MatrixXd::Random(10, 1)/2;

    streamoff read_position = 0;
    read_position = read(&wab.W1, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read_position = read(&wab.B1, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read_position = read(&wab.W2, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read_position = read(&wab.B2, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read_position = read(&wab.W3, read_position, WEIGHTS_AND_BIASES_FILE_PATH);
    read(&wab.B3, read_position, WEIGHTS_AND_BIASES_FILE_PATH);

    // Do forward propagation with the stored weights and biases
    states_and_activations fp = forward_prop(X, wab);

    // Get the number of correct predictions
    int count = get_num_correct(get_predictions(fp.A3, NUM_TEST_IMAGES), Y, NUM_TEST_IMAGES);

    // Optionally print out the test labels and images
    if (PRINT_LABELS_AND_IMAGES)
        print_batch(X, Y, NUM_TEST_IMAGES);

    // Print the accuracy of the trained neural network
    cout << "Accuracy: " << count << "/" << NUM_TEST_IMAGES << "\n";

    return 0;
}