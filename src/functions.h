#ifndef DIGIT_RECOGNITION_FUNCTIONS_H
#define DIGIT_RECOGNITION_FUNCTIONS_H

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"
#define WEIGHTS_AND_BIASES_FILE_PATH R"(.\src\wandb.bin)"
#define LABEL_START 8
#define IMAGE_START 16
#define BATCH_SIZE 32
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.01
#define L1_SIZE 350
#define NUM_EPOCHS 1000
#define ACTIVATION_FUNCTION TANH
#define PRINT_IMAGE_AND_LABEL false

#include <Eigen/Dense>

typedef struct {
    Eigen::MatrixXd Z1, A1, Z2, A2;
} fp_return;

typedef struct {
    Eigen::MatrixXd dZ2, dW2, dB2, dZ1, dW1, dB1;
} bp_return;

enum Activation {TANH= 0, RELU= 1};

std::streamoff save(const Eigen::MatrixXd &X, std::streamoff position);

std::streamoff read(Eigen::MatrixXd *X, std::streamoff position);

Eigen::MatrixXd get_labels(int offset, int size, std::string path);

Eigen::MatrixXd get_images(int offset, int size, std::string path);

void print_batch(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, int size);

Eigen::MatrixXd softmax(const Eigen::MatrixXd &Z);

Eigen::MatrixXd deriv_tanh(Eigen::MatrixXd Z);

double ReLU(double x);

double deriv_ReLU(double x);

fp_return forward_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &W1, const Eigen::MatrixXd &B1,
                       const Eigen::MatrixXd &W2, const Eigen::MatrixXd &B2);

bp_return back_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::MatrixXd &Z1,
                    const Eigen::MatrixXd &A1, const Eigen::MatrixXd &A2, const Eigen::MatrixXd &W2);

void update_params(Eigen::MatrixXd *W1, Eigen::MatrixXd *B1, Eigen::MatrixXd *W2, Eigen::MatrixXd *B2,
                   const Eigen::MatrixXd &dW1, const Eigen::MatrixXd &dB1, const Eigen::MatrixXd &dW2,
                   const Eigen::MatrixXd &dB2, double learning_rate);

Eigen::MatrixXd get_predictions(const Eigen::MatrixXd &AL, int size);

int get_num_correct(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Y, int size);

void gradient_descent(Eigen::MatrixXd *W1, Eigen::MatrixXd *B1, Eigen::MatrixXd *W2, Eigen::MatrixXd *B2, double learning_rate, int epoch);

#endif
