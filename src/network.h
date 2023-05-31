#ifndef DIGIT_RECOGNITION_NETWORK_H
#define DIGIT_RECOGNITION_NETWORK_H

#define TRAIN_LABELS_FILE_PATH R"(C:\Users\minsi\Coding Projects\GitHub Synced\Digit-Recognition\data\train-labels.idx1-ubyte)"
#define TRAIN_IMAGES_FILE_PATH R"(C:\Users\minsi\Coding Projects\GitHub Synced\Digit-Recognition\data\train-images.idx3-ubyte)"
#define TEST_LABELS_FILE_PATH R"(C:\Users\minsi\Coding Projects\GitHub Synced\Digit-Recognition\data\t10k-labels.idx1-ubyte)"
#define TEST_IMAGES_FILE_PATH R"(C:\Users\minsi\Coding Projects\GitHub Synced\Digit-Recognition\data\t10k-images.idx3-ubyte)"
#define WEIGHTS_AND_BIASES_FILE_PATH R"(C:\Users\minsi\Coding Projects\GitHub Synced\Digit-Recognition\src\wandb.bin)"
#define LABEL_START 8
#define IMAGE_START 16
#define BATCH_SIZE 32
#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define LEARNING_RATE 0.01
#define L1_SIZE 300
#define NUM_EPOCHS 450
#define ACTIVATION_FUNCTION TANH
#define PRINT_IMAGE_AND_LABEL false

typedef struct {
    Eigen::MatrixXd Z1, A1, Z2, A2;
} fp_return;

typedef struct {
    Eigen::MatrixXd dZ2, dW2, dB2, dZ1, dW1, dB1;
} bp_return;

enum Activation {TANH= 0, RELU= 1};

#endif
