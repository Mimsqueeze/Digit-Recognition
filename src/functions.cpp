#include <fstream>
#include <iostream>
#include "functions.h"
#include <Eigen/Dense>
#include <cfloat>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

streamoff save(const MatrixXd &X, streamoff position) {
    // Get number of rows and columns
    int rows = X.rows();
    int cols = X.cols();

    ofstream file;
    // Open file
    if (position == 0) {
        file = ofstream(WEIGHTS_AND_BIASES_FILE_PATH, ios::out | ios::binary);
    } else {
        file = ofstream(WEIGHTS_AND_BIASES_FILE_PATH, ios::app | ios::binary);
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

streamoff read(MatrixXd *X, streamoff position) {
    // Get number of rows and columns
    int rows = (*X).rows();
    int cols = (*X).cols();

    // Open file
    ifstream file(WEIGHTS_AND_BIASES_FILE_PATH, ios::in | ios::binary);

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


fp_return
forward_prop(const MatrixXd &X, const MatrixXd &W1, const MatrixXd &B1, const MatrixXd &W2, const MatrixXd &B2) {
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

void gradient_descent(MatrixXd *W1, MatrixXd *B1, MatrixXd *W2, MatrixXd *B2, double learning_rate, int epoch) {
    MatrixXd dW1 = MatrixXd::Zero(L1_SIZE, 784);
    MatrixXd dB1 = MatrixXd::Zero(L1_SIZE, 1);
    MatrixXd dW2 = MatrixXd::Zero(10, L1_SIZE);
    MatrixXd dB2 = MatrixXd::Zero(10, 1);
    int count = 0;

    for (int batch_offset = 0; batch_offset < NUM_TRAIN_IMAGES; batch_offset += BATCH_SIZE) {
        MatrixXd X = get_images(batch_offset * 784, BATCH_SIZE, TRAIN_IMAGES_FILE_PATH);
        MatrixXd Y = get_labels(batch_offset, BATCH_SIZE, TRAIN_LABELS_FILE_PATH);
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

    dW1 /= BATCH_SIZE;
    dB1 /= BATCH_SIZE;
    dW2 /= BATCH_SIZE;
    dB2 /= BATCH_SIZE;

    update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate);
    cout << "Epoch: " << epoch << "\n";
    cout << "Accuracy: " << count << "/" << NUM_TRAIN_IMAGES << "\n";
}