#ifndef DIGIT_RECOGNITION_CONV_NETWORK_H
#define DIGIT_RECOGNITION_CONV_NETWORK_H

#define WEIGHTS_AND_BIASES_FILE_PATH R"(.\src\convolutional network\wandb.bin)"
#define CONV_IMAGES_FILE_PATH R"(.\src\convolutional network\conv-train-images.bin)"
#define L1_SIZE 50
#define POOLING_WINDOW_SIZE 2
#define POOLING_STRIDE_SIZE 2
#define POOLING_OUTPUT_SIZE (((26 % POOLING_STRIDE_SIZE) == 0) ? (26/POOLING_STRIDE_SIZE) : (26/POOLING_STRIDE_SIZE + 1))
#define CONVOLUTION_OUTPUT_SIZE (POOLING_OUTPUT_SIZE*POOLING_OUTPUT_SIZE*4)

#include <Eigen/Dense>

typedef struct {
    Eigen::MatrixXd Z1, A1, Z2, A2;
} states_and_activations;

typedef struct {
    Eigen::MatrixXd dZ2, dW2, dB2, dZ1, dW1, dB1;
} derivatives;

states_and_activations forward_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &W1, const Eigen::MatrixXd &B1,
                                    const Eigen::MatrixXd &W2, const Eigen::MatrixXd &B2);

derivatives back_prop(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const Eigen::MatrixXd &Z1,
                      const Eigen::MatrixXd &A1, const Eigen::MatrixXd &Z2, const Eigen::MatrixXd &A2,
                      const Eigen::MatrixXd &W2);

void update_params(Eigen::MatrixXd *W1, Eigen::MatrixXd *B1, Eigen::MatrixXd *W2, Eigen::MatrixXd *B2,
                   const Eigen::MatrixXd &dW1, const Eigen::MatrixXd &dB1, const Eigen::MatrixXd &dW2,
                   const Eigen::MatrixXd &dB2, double learning_rate);

int gradient_descent(Eigen::MatrixXd *W1, Eigen::MatrixXd *B1, Eigen::MatrixXd *W2, Eigen::MatrixXd *B2,
                     double learning_rate, int epoch);

Eigen::MatrixXd getConvolution(const Eigen::MatrixXd &X);

Eigen::MatrixXd convolve(const Eigen::MatrixXd &A, const Eigen::MatrixXd &filter);

Eigen::MatrixXd pool(const Eigen::MatrixXd &A);

Eigen::MatrixXd convertMatrixToCol(const Eigen::MatrixXd &matrix);

Eigen::MatrixXd convertColToMatrix(const Eigen::MatrixXd &column);

Eigen::MatrixXd combineColumns(const std::vector<Eigen::MatrixXd> &column_list);

void insertColumn(Eigen::MatrixXd &result, const Eigen::MatrixXd &column, int col_num);

void conv_print_batch(const Eigen::MatrixXd &C, const Eigen::MatrixXd &Y, int size);

Eigen::MatrixXd conv_get_image_batch(const int offsets[], int index, int size, const std::string &path);

#endif //DIGIT_RECOGNITION_CONV_NETWORK_H
