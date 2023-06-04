#ifndef DIGIT_RECOGNITION_FUNCTIONS_H
#define DIGIT_RECOGNITION_FUNCTIONS_H

#define TRAIN_LABELS_FILE_PATH R"(.\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(.\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(.\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(.\data\t10k-images.idx3-ubyte)"

#define LABEL_START 8
#define IMAGE_START 16
#define BATCH_SIZE 32

#define NUM_TRAIN_IMAGES 10000
#define NUM_BATCHES (NUM_TRAIN_IMAGES/BATCH_SIZE)
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.1
#define NUM_EPOCHS 50
#define ACTIVATION_FUNCTION TANH

#define PRINT_IMAGE_AND_LABEL false

#include <Eigen/Dense>

enum Activation {
    TANH = 0, RELU = 1, LEAKY_RELU = 2
};

std::streamoff save(const Eigen::MatrixXd &X, std::streamoff position, std::string path);

std::streamoff read(Eigen::MatrixXd *X, std::streamoff position, std::string path);

Eigen::MatrixXd get_labels(int offset, int size, std::string path);

Eigen::MatrixXd get_images(int offset, int size, std::string path);

Eigen::MatrixXd get_label_batch(int arr[], int index, int size, std::string path);

Eigen::MatrixXd get_image_batch(int arr[], int index, int size, std::string path);

void print_batch(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, int size);

Eigen::MatrixXd softmax(const Eigen::MatrixXd &Z);

Eigen::MatrixXd deriv_tanh(Eigen::MatrixXd Z);

double ReLU(double x);

double deriv_ReLU(double x);

double leaky_ReLU(double x);

double deriv_leaky_ReLU(double x);

Eigen::MatrixXd get_predictions(const Eigen::MatrixXd &AL, int size);

int get_num_correct(const Eigen::MatrixXd &P, const Eigen::MatrixXd &Y, int size);

#endif
