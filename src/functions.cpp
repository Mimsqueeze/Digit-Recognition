#include <fstream>
#include <iostream>
#include "functions.h"
#include <Eigen/Dense>
#include <cfloat>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

streamoff save(const MatrixXd &X, streamoff position, const string &path) {
    // Get number of rows and columns
    int rows = (int) X.rows();
    int cols = (int) X.cols();

    // Declare file
    ofstream file;

    // Open file
    if (position == 0) {
        file = ofstream(path, ios::out | ios::binary);
    } else {
        file = ofstream(path, ios::app | ios::binary);
    }

    if (file.is_open()) {
        // Save matrix X into the offset position
        file.seekp(position);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                file.write((char *) &X(i, j), sizeof(double));
            }
        }
        // Save the resulting position
        position = file.tellp();

        // Close the file
        file.close();
    } else {
        cout << "Error: Failed to open file WANDB";
        exit(1);
    }

    return position;
}

streamoff read(MatrixXd *X, streamoff position, const string &path) {
    // Get number of rows and columns
    int rows = (int) (*X).rows();
    int cols = (int) (*X).cols();

    // Open file
    ifstream file(path, ios::in | ios::binary);

    if (file.is_open()) {
        // Extract matrix X from offset position
        file.seekg(position);

        double temp = 0;
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                file.read((char *) &temp, sizeof(double));
                (*X)(i, j) = temp;
            }
        }
        // Save the resulting position
        position = file.tellg();

        // Close the file
        file.close();
    } else {
        cout << "Error: Failed to open file WANDB";
        exit(1);
    }

    return position;
}

MatrixXd get_labels(int offset, int size, const string &path) {
    // Create Y Matrix of dimension 10 x size
    MatrixXd Y = MatrixXd::Zero(10, size);

    // Open file
    ifstream labels_file(path, ios::in | ios::binary);

    if (labels_file.is_open()) {
        // Extract matrix Y by reading size number of labels from offset of beginning of the file
        labels_file.seekg(LABEL_START + offset);
        int temp = 0;
        for (int i = 0; i < size; i++) {
            labels_file.read((char *) &temp, 1);
            Y(temp, i) = 1;
        }
        // Close the file
        labels_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return Y;
}

Eigen::MatrixXd get_images(int offset, int size, const string &path) {
    // Create X Matrix of dimension 784 x size to represent input layer
    MatrixXd X = MatrixXd::Zero(784, size);

    // Open file
    ifstream images_file(path, ios::in | ios::binary);

    if (images_file.is_open()) {
        // Extract matrix X by reading size number of images from offset of beginning of the file
        images_file.seekg(IMAGE_START + offset);
        int temp = 0;
        for (int i = 0; i < 784 * size; i++) {
            images_file.read((char *) &temp, 1);

            // Transform temp from range [0, 255] to range [-1, 1]
            double transform= (temp-127.5)/127.5;

            X(i % 784, i / 784) = transform;
        }
        // Close the file
        images_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return X;
}

void print_batch(const MatrixXd &X, const MatrixXd &Y, int size) {
    // For size number of labels/images, print them
    for (int i = 0; i < size; i++) {
        // Print label
        cout << "The following number is: ";
        for (int j = 0; j < 10; j++) {
            if (Y(j, i) == 1) {
                cout << j << "\n";
                break;
            }
        }
        // Print image
        for (int j = 0; j < 784; j++) {
            if (j != 0 && j % 28 == 0) {
                cout << "\n";
            }
            if (X(j, i) < 0) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << "\n";
    }
}

MatrixXd softmax(const MatrixXd &Z) {
    // Convert into array
    MatrixXd Z1 = Z.array();

    // Find max values of each column
    VectorXd Max = Z1.colwise().maxCoeff();

    // Subtract max, compute exponential, compute sum, and then compute logarithm
    MatrixXd Z2 = (Z1.rowwise() - Max.transpose()).array().exp().colwise().sum().array().log();

    // Compute offset
    VectorXd Offset = Z2.transpose() + Max;

    // Subtract offset and compute exponential
    return (Z1.rowwise() - Offset.transpose()).array().exp();
}

MatrixXd deriv_tanh(const MatrixXd &Z) {
    return 1 - Z.array().tanh().pow(2);
}

double ReLU(double x) {
    if (x > 0)
        return x;
    else
        return 0;
}

double deriv_ReLU(double x) {
    return x > 0;
}

double leaky_ReLU(double x) {
    if (x > 0)
        return x;
    else
        return 0.01 * x;
}

double deriv_leaky_ReLU(double x) {
    if (x > 0)
        return 1;
    else
        return 0.01;
}

MatrixXd get_predictions(const MatrixXd &AL, int size) {
    // Initialize matrix of predictions
    MatrixXd P = MatrixXd::Zero(10, size);

    // For each column of AL, find its largest value and fill its position in P 1. Leave the rest as 0.
    // Essentially taking the argmax to find the prediction
    for (int i = 0; i < size; i++) {
        double largest = -DBL_MAX;
        int prediction = -1;
        for (int j = 0; j < 10; j++) {
            if (AL(j, i) > largest) {
                prediction = j;
                largest = AL(j, i);
            }
        }
        P(prediction, i) = 1;
    }

    return P;
}

int get_num_correct(const MatrixXd &P, const MatrixXd &Y, int size) {
    // Initialize variable to store number of correct predictions
    int correct = 0;

    // For size number of columns, compare position of 1's. If they match, it's a correct prediction.
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < 10; j++) {
            if (P(j, i) == 1) {
                if (Y(j, i) == 1)
                    correct++;
                break;
            }
        }
    }

    return correct;
}

MatrixXd get_label_batch(const int offsets[], int index, int size, const string &path) {
    // Create Y Matrix of dimension 10 x size
    MatrixXd Y = MatrixXd::Zero(10, size);

    // Open file
    ifstream labels_file(path, ios::in | ios::binary);

    if (labels_file.is_open()) {
        // Extract size number of random labels
        for (int i = 0; i < size; i++) {
            labels_file.seekg(LABEL_START + offsets[index + i]);
            int temp = 0;
            labels_file.read((char *) &temp, 1);
            Y(temp, i) = 1;
        }

        // Close the file
        labels_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return Y;
}

MatrixXd get_image_batch(const int offsets[], int index, int size, const string &path) {
    // Create X Matrix of dimension 784 x size to represent input layer
    MatrixXd X = MatrixXd::Zero(784, size);

    // Open file
    ifstream images_file(path, ios::in | ios::binary);

    if (images_file.is_open()) {
        // Extract size number of random images
        for (int i = 0; i < size; i++) {
            images_file.seekg(IMAGE_START + 784 * offsets[index + i]);
            for (int j= 0; j < 784; j++) {
                int temp = 0;
                images_file.read((char *) &temp, 1);

                // Transform temp from range [0, 255] to range [-1, 1]
                double transform= (temp-127.5)/127.5;

                X(j % 784, i) = transform;
            }
        }
        // Close the file
        images_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return X;
}