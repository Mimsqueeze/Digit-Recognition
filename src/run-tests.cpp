#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <cfloat>
#include "network.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

streamoff read(MatrixXd *X, streamoff position);
MatrixXd get_test_labels();
MatrixXd get_test_images();
fp_return forward_prop(const MatrixXd &X, const MatrixXd &W1, const MatrixXd &B1, const MatrixXd &W2, const MatrixXd &B2);
MatrixXd get_predictions(const MatrixXd &AL);
int get_num_correct(const MatrixXd &P, const MatrixXd &Y);
void print_batch(const MatrixXd &X, const MatrixXd &Y);

int activation= ACTIVATION_FUNCTION;

int main() {
    MatrixXd X= get_test_images();
    MatrixXd Y= get_test_labels();
    MatrixXd W1(L1_SIZE, 784);
    MatrixXd B1(L1_SIZE, 1);
    MatrixXd W2(10, L1_SIZE);
    MatrixXd B2(10, 1);

    streamoff read_position= 0;
    read_position= read(&W1, read_position);
    read_position= read(&B1, read_position);
    read_position= read(&W2, read_position);
    read(&B2, read_position);

    fp_return fp= forward_prop(X, W1, B1, W2, B2);
    int count= get_num_correct(get_predictions(fp.A2), Y);

    if (PRINT_IMAGE_AND_LABEL)
        print_batch(X, Y);

    cout << "Accuracy: " << count << "/" << NUM_TEST_IMAGES << "\n";

    return 0;
}

void print_batch(const MatrixXd &X, const MatrixXd &Y) {
    for (int i= 0; i < BATCH_SIZE; i++) {
        cout << "The following number is: ";
        for (int j= 0; j < 10; j++) {
            if (Y(j,i) == 1) {
                cout << j << "\n";
                break;
            }
        }

        for (int j= 0; j < 784; j++) {
            if (j != 0 && j % 28 == 0) {
                cout << "\n";
            }
            if (X(j,i) < 128) {
                cout << "@.@";
            } else {
                cout << " . ";
            }
        }
        cout << "\n";
    }
}

MatrixXd get_test_labels() {
    // Create Y Matrix of dimension 10 x BATCH_SIZE
    MatrixXd Y= MatrixXd::Zero(10, NUM_TEST_IMAGES);

    // Open file
    ifstream train_labels_file(TEST_LABELS_FILE_PATH, ios::in | ios::binary);

    if (train_labels_file.is_open()) {
        train_labels_file.seekg(LABEL_START);
        int temp= 0;
        for (int i= 0; i < NUM_TEST_IMAGES; i++) {
            train_labels_file.read((char*)&temp, 1);
            Y(temp, i)= 1;
        }
        train_labels_file.close();
    } else {
        cout << "Error: Failed to open file TRAIN_LABELS";
        exit(1);
    }

    return Y;
}

MatrixXd get_test_images() {
    // Create X Matrix of dimension 784 x BATCH_SIZE to represent input layer
    MatrixXd X= MatrixXd::Zero(784, NUM_TEST_IMAGES);

    // Open file
    ifstream train_images_file(TEST_IMAGES_FILE_PATH, ios::in | ios::binary);

    if (train_images_file.is_open()) {
        train_images_file.seekg(IMAGE_START);
        int temp= 0;
        for (int i= 0; i < 784*NUM_TEST_IMAGES; i++) {
            train_images_file.read((char*)&temp, 1);
            X(i%784, i/784)= temp;
        }
        train_images_file.close();
    } else {
        cout << "Error: Failed to open file TEST_IMAGES";
        exit(1);
    }

    return X;
}

streamoff read(MatrixXd *X, streamoff position) {
    // Get number of rows and columns
    int rows= (*X).rows();
    int cols= (*X).cols();

    // Open file
    ifstream file(WEIGHTS_AND_BIASES_FILE_PATH, ios::in | ios::binary);

    if (file.is_open()) {
        file.seekg(position);
        double temp= 0;
        for (int i= 0; i < rows; i++) {
            for (int j= 0; j < cols; j++) {
                file.read((char *) &temp, sizeof(double));
                (*X)(i,j)= temp;
            }
        }
        position= file.tellg();
        file.close();
    } else {
        cout << "Error: Failed to open file WANDB";
        exit(1);
    }

    return position;
}

MatrixXd get_predictions(const MatrixXd &AL) {
    MatrixXd P= MatrixXd::Zero(10, NUM_TEST_IMAGES);
    for (int i= 0; i < NUM_TEST_IMAGES; i++) {
        double largest= -DBL_MAX;
        int prediction= -1;
        for (int j= 0; j < 10; j++) {
            if(AL(j,i) > largest) {
                prediction= j;
                largest= AL(j,i);
            }
        }
        P(prediction, i)= 1;
    }

    return P;
}

int get_num_correct(const MatrixXd &P, const MatrixXd &Y) {
    int correct= 0;
    for (int i= 0; i < NUM_TEST_IMAGES; i++) {
        for (int j= 0; j < 10; j++) {
            if (P(j,i) == 1) {
                if (Y(j,i) == 1)
                    correct++;
                break;
            }
        }
    }

    return correct;
}

MatrixXd softmax(const MatrixXd &Z) {
    // Convert into array
    MatrixXd Z1= Z.array();

    // Find max values of each column
    VectorXd Max= Z1.colwise().maxCoeff();

    // Subtract max, compute exponential, compute sum, and then compute logarithm
    MatrixXd Z2= (Z1.rowwise() - Max.transpose()).array().exp().colwise().sum().array().log();

    // Compute offset
    VectorXd Offset= Z2.transpose() + Max;

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

fp_return forward_prop(const MatrixXd &X, const MatrixXd &W1, const MatrixXd &B1, const MatrixXd &W2, const MatrixXd &B2) {
    fp_return fp;
    fp.Z1= W1*X + B1*MatrixXd::Ones(1, NUM_TEST_IMAGES);
    if (activation == TANH)
        fp.A1= fp.Z1.array().tanh();
    else if (activation == RELU)
        fp.A1= fp.Z1.array().unaryExpr(&ReLU);
    fp.Z2= W2*fp.A1 + B2*MatrixXd::Ones(1, NUM_TEST_IMAGES);
    fp.A2= softmax(fp.Z2);

    return fp;
}