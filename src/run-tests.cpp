#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "functions.h"
#include "functions.cpp"

using namespace std;
using Eigen::MatrixXd;

int main() {
    MatrixXd X = get_images(0, NUM_TEST_IMAGES, TEST_IMAGES_FILE_PATH);
    MatrixXd Y = get_labels(0, NUM_TEST_IMAGES, TEST_LABELS_FILE_PATH);
    MatrixXd W1(L1_SIZE, 784);
    MatrixXd B1(L1_SIZE, 1);
    MatrixXd W2(10, L1_SIZE);
    MatrixXd B2(10, 1);

    streamoff read_position = 0;
    read_position = read(&W1, read_position);
    read_position = read(&B1, read_position);
    read_position = read(&W2, read_position);
    read(&B2, read_position);

    fp_return fp = forward_prop(X, W1, B1, W2, B2);
    int count = get_num_correct(get_predictions(fp.A2, NUM_TEST_IMAGES), Y, NUM_TEST_IMAGES);

    if (PRINT_IMAGE_AND_LABEL)
        print_batch(X, Y, NUM_TEST_IMAGES);

    cout << "Accuracy: " << count << "/" << NUM_TEST_IMAGES << "\n";

    return 0;
}