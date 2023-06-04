#include <fstream>
#include <iostream>
#include "functions.h"
#include <Eigen/Dense>
#include <cfloat>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

streamoff save(const MatrixXd &X, streamoff position, string path) {
    // Get number of rows and columns
    int rows = X.rows();
    int cols = X.cols();

    ofstream file;
    // Open file
    if (position == 0) {
        file = ofstream(path, ios::out | ios::binary);
    } else {
        file = ofstream(path, ios::app | ios::binary);
    }

    if (file.is_open()) {
        file.seekp(position);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file.write((char *) &X(i, j), sizeof(double));
            }
        }
        position = file.tellp();
        file.close();
    } else {
        cout << "Error: Failed to open file WANDB";
        exit(1);
    }

    return position;
}

streamoff read(MatrixXd *X, streamoff position, string path) {
    // Get number of rows and columns
    int rows = (*X).rows();
    int cols = (*X).cols();

    // Open file
    ifstream file(path, ios::in | ios::binary);

    if (file.is_open()) {
        file.seekg(position);
        double temp = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                file.read((char *) &temp, sizeof(double));
                (*X)(i, j) = temp;
            }
        }
        position = file.tellg();
        file.close();
    } else {
        cout << "Error: Failed to open file WANDB";
        exit(1);
    }

    return position;
}

MatrixXd get_labels(int offset, int size, string path) {
    // Create Y Matrix of dimension 10 x size
    MatrixXd Y = MatrixXd::Zero(10, size);

    // Open file
    ifstream labels_file(path, ios::in | ios::binary);

    if (labels_file.is_open()) {
        labels_file.seekg(LABEL_START + offset);
        int temp = 0;
        for (int i = 0; i < size; i++) {
            labels_file.read((char *) &temp, 1);
            Y(temp, i) = 1;
        }
        labels_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return Y;
}

Eigen::MatrixXd get_images(int offset, int size, string path) {
    // Create X Matrix of dimension 784 x size to represent input layer
    MatrixXd X = MatrixXd::Zero(784, size);

    // Open file
    ifstream images_file(path, ios::in | ios::binary);

    if (images_file.is_open()) {
        images_file.seekg(IMAGE_START + offset);
        int temp = 0;
        for (int i = 0; i < 784 * size; i++) {
            images_file.read((char *) &temp, 1);
            X(i % 784, i / 784) = temp;
        }
        images_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return X;
}

void print_batch(const MatrixXd &X, const MatrixXd &Y, int size) {
    for (int i = 0; i < size; i++) {
        cout << "The following number is: ";
        for (int j = 0; j < 10; j++) {
            if (Y(j, i) == 1) {
                cout << j << "\n";
                break;
            }
        }

        for (int j = 0; j < 784; j++) {
            if (j != 0 && j % 28 == 0) {
                cout << "\n";
            }
            if (X(j, i) < 128) {
                cout << "@.@";
            } else {
                cout << " . ";
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

MatrixXd deriv_tanh(MatrixXd Z) {
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
    MatrixXd P = MatrixXd::Zero(10, size);
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
    int correct = 0;
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

MatrixXd get_label_batch(int arr[], int index, int size, string path) {
    // Create Y Matrix of dimension 10 x size
    MatrixXd Y = MatrixXd::Zero(10, size);

    // Open file
    ifstream labels_file(path, ios::in | ios::binary);

    if (labels_file.is_open()) {
        for (int i = 0; i < size; i++) {
            labels_file.seekg(LABEL_START + arr[index + i]);
            int temp = 0;
            labels_file.read((char *) &temp, 1);
            Y(temp, i) = 1;
        }
        labels_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return Y;
}

MatrixXd get_image_batch(int arr[], int index, int size, string path) {
    // Create X Matrix of dimension 784 x size to represent input layer
    MatrixXd X = MatrixXd::Zero(784, size);

    // Open file
    ifstream images_file(path, ios::in | ios::binary);

    if (images_file.is_open()) {
        for (int i = 0; i < size; i++) {
            images_file.seekg(IMAGE_START + 784*arr[index + i]);
            for (int j= 0; j < 784; j++) {
                int temp = 0;
                images_file.read((char *) &temp, 1);
                X(j % 784, i) = temp;
            }
        }
        images_file.close();
    } else {
        cout << "Error: Failed to open file " << path << endl;
        exit(1);
    }

    return X;
}

