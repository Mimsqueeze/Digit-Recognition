#include <fstream>
#include <iostream>
#include "conv-network.h"
#include "../functions.h"
#include <Eigen/Dense>
#include <random>
#include <chrono>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

states_and_activations forward_prop(const MatrixXd &X, const MatrixXd &W1, const MatrixXd &B1, const MatrixXd &W2,
                                    const MatrixXd &B2) {
    // Initialize return struct
    states_and_activations fp;

    // Neuron state:
    // For all neurons in layer l, its state is defined as its bias plus all incoming connections
    // (weight * activation from all source neurons)
    // Zl = Wl * Al-1 + Bl

    // Neuron activation:
    // For all neurons in layer l, its activation is defined as its state passed through an activation function f
    // Al = f(Zl)

    // Layer 1 (Hidden layer)
    fp.Z1 = W1 * X + B1 * MatrixXd::Ones(1, BATCH_SIZE);
    if (ACTIVATION_FUNCTION == TANH)
        fp.A1 = fp.Z1.array().tanh();
    else if (ACTIVATION_FUNCTION == RELU)
        fp.A1 = fp.Z1.array().unaryExpr(&ReLU);
    else if (ACTIVATION_FUNCTION == LEAKY_RELU)
        fp.A1 = fp.Z1.array().unaryExpr(&leaky_ReLU);

    // Layer 2 (Output layer)
    fp.Z2 = W2 * fp.A1 + B2 * MatrixXd::Ones(1, BATCH_SIZE);
    fp.A2 = softmax(fp.Z2);

    return fp;
}

derivatives back_prop(const MatrixXd &X, const MatrixXd &Y, const MatrixXd &Z1, const MatrixXd &A1, const MatrixXd &Z2,
                      const MatrixXd &A2, const MatrixXd &W2) {
    // Initialize return struct
    derivatives bp;

    // Cost function:
    // C = 1/2(Y - AL)^2

    // Change of cost relative to change in state, where L is the last layer:
    // dC/dZL = dC/dAL             (*) f'(zL)
    // dC/dZl = (Wl+1)T * dC/dZl+1 (*) f'(zl)

    // Change of cost relative to change in weights
    // dC/dWl = Al-1 * dC/dZl

    // Change of cost relative to change in biases
    // dC/dBl = dC/dZl

    // Compute derivatives for Layer 2 (Output layer)
    bp.dZ2 = A2 - Y;
    bp.dW2 = (1.0 / BATCH_SIZE) * bp.dZ2 * A1.transpose();
    bp.dB2 = (1.0 / BATCH_SIZE) * bp.dZ2.rowwise().sum();

    // Computes derivatives for Layer 1 (Hidden layer)
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
    // Update the weights and biases by moving in the direction of the steepest descent
    *W1 = *W1 - learning_rate * dW1;
    *B1 = *B1 - learning_rate * dB1;
    *W2 = *W2 - learning_rate * dW2;
    *B2 = *B2 - learning_rate * dB2;
}

int gradient_descent(MatrixXd *W1, MatrixXd *B1, MatrixXd *W2, MatrixXd *B2, double learning_rate, int epoch) {\
    // Initialize derivative matrices to 0.
    // Note, these are basically storing the "nudges" that will be done to W1, B1, W2, and B2
    MatrixXd dW1 = MatrixXd::Zero(L1_SIZE, CONVOLUTION_OUTPUT_SIZE);
    MatrixXd dB1 = MatrixXd::Zero(L1_SIZE, 1);
    MatrixXd dW2 = MatrixXd::Zero(10, L1_SIZE);
    MatrixXd dB2 = MatrixXd::Zero(10, 1);

    // Initialize count variable to store number of correct predictions
    int count = 0;

    // Create array of offsets each associated with a label/image pair
    int data_offsets[NUM_TRAIN_IMAGES];

    // Fill with numbers 0 to 1-NUM_TRAIN_IMAGES in increasing order
    iota(data_offsets, data_offsets + NUM_TRAIN_IMAGES, 0);

    // Randomly shuffle array of offsets, to randomize image selection in mini-batches
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(data_offsets, data_offsets + NUM_TRAIN_IMAGES, std::default_random_engine(seed));

    // For every training image, go through gradient descent in mini-batches
    for (int i = 0; i < NUM_TRAIN_IMAGES; i += BATCH_SIZE) {
        // Get convolved image and label batch
        MatrixXd C = conv_get_image_batch(data_offsets, i, BATCH_SIZE, CONV_IMAGES_FILE_PATH);
        MatrixXd Y = get_label_batch(data_offsets, i, BATCH_SIZE, TRAIN_LABELS_FILE_PATH);

        // Optionally print out the training labels and images
        if (PRINT_LABELS_AND_IMAGES)
            conv_print_batch(C, Y, BATCH_SIZE);

        // Forward propagate to get Z1, A1, Z2, and A2
        states_and_activations fp = forward_prop(C, *W1, *B1, *W2, *B2);

        // Back propagate to get dW1, dB1, dW2, dB2
        derivatives bp = back_prop(C, Y, fp.Z1, fp.A1, fp.Z2, fp.A2, *W2);

        // Add derivatives from mini-batch, in other words add the "nudges"
        dW1 += bp.dW1;
        dB1 += bp.dB1;
        dW2 += bp.dW2;
        dB2 += bp.dB2;

        // Add the number of correct predictions from the mini-batch
        count += get_num_correct(get_predictions(fp.A2, BATCH_SIZE), Y, BATCH_SIZE);
    }

    // Divide each derivative or "nudge" value by the number of batches to find the average among all batches
    dW1 /= NUM_BATCHES;
    dB1 /= NUM_BATCHES;
    dW2 /= NUM_BATCHES;
    dB2 /= NUM_BATCHES;

    // Update the parameters W1, B1, W2, and B2 with the "nudges"
    update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate);

    return count;
}

MatrixXd getConvolution(const MatrixXd &X) {
    int r= (int) sqrt(X.rows()) - 2;
    int n= ((r % POOLING_STRIDE_SIZE) == 0) ? (r/POOLING_STRIDE_SIZE) : (r/POOLING_STRIDE_SIZE + 1);
    MatrixXd result(n*n*4, X.cols());

    MatrixXd vertical_filter (3, 3);
    vertical_filter <<
    -0.25,      0,  0.25,
    -0.50,      0,  0.50,
    -0.25,      0,  0.25;

    MatrixXd horizontal_filter (3, 3);
    horizontal_filter <<
    -0.25,  -0.50, -0.25,
        0,      0,     0,
     0.25,   0.50,  0.25;

    MatrixXd neg_diag_filter (3, 3);
    neg_diag_filter <<
        0,  -0.25, -0.50,
     0.25,      0, -0.25,
     0.50,   0.25,     0;

    MatrixXd pos_diag_filter (3, 3);
    pos_diag_filter <<
     0.50,   0.25,     0,
     0.25,      0, -0.25,
        0,  -0.25, -0.50;

    for (int i= 0; i < X.cols(); i++) {
        MatrixXd image= convertColToMatrix(X.col(i));

        MatrixXd vertical_convolution= convolve(image, vertical_filter);
        MatrixXd pool_vertical= pool(vertical_convolution);
        MatrixXd vertical_column= convertMatrixToCol(pool_vertical);

        MatrixXd horizontal_convolution= convolve(image, horizontal_filter);
        MatrixXd pool_horizontal= pool(horizontal_convolution);
        MatrixXd horizontal_column= convertMatrixToCol(pool_horizontal);

        MatrixXd pos_diag_convolution= convolve(image, pos_diag_filter);
        MatrixXd pool_pos_diag= pool(pos_diag_convolution);
        MatrixXd pos_diag_column= convertMatrixToCol(pool_pos_diag);

        MatrixXd neg_diag_convolution= convolve(image, neg_diag_filter);
        MatrixXd pool_neg_diag= pool(neg_diag_convolution);
        MatrixXd neg_diag_column= convertMatrixToCol(pool_neg_diag);

        vector<MatrixXd> column_list= {vertical_column, horizontal_column, pos_diag_column, neg_diag_column};
        MatrixXd column= combineColumns(column_list);
        insertColumn(result, column, i);
    }

    return result;
}

MatrixXd pool(const MatrixXd &A) {
    int r= (int) A.rows();
    int n= ((r % POOLING_STRIDE_SIZE) == 0) ? (r/POOLING_STRIDE_SIZE) : (r/POOLING_STRIDE_SIZE + 1);
    MatrixXd result(n, n);

    // Coordinates of resulting matrix
    int x= 0, y= 0;

    // Fill the resulting matrix
    while (y < n) {

        // Define a patch of POOLING_WINDOW_SIZExPOOLING_WINDOW_SIZE pixels of A
        vector<double> patch;

        // Fill in the patch
        for (int i= 0; i < POOLING_WINDOW_SIZE; i++) {
            for (int j= 0; j < POOLING_WINDOW_SIZE; j++) {
                int a= POOLING_STRIDE_SIZE*x + i;
                int b= POOLING_STRIDE_SIZE*y + j;
                if ((a < r) && (b < r)) {
                    patch.push_back(A(a, b));
                }
            }
        }
        double maximum= *max_element(patch.begin(), patch.end());
        result(x, y)= maximum;

        if (++x == n) {
            x= 0;
            y++;
        }
    }

    return result;
}

MatrixXd combineColumns(const vector<MatrixXd> &column_list) {
    int num_columns= (int) column_list.size();
    int total_elts= 0;
    vector<int> num_elts_list(num_columns);
    for (int i= 0; i < num_columns; i++) {
        total_elts+= num_elts_list[i]= (int) column_list[i].rows();
    }
    MatrixXd result(total_elts, 1);
    int z= 0;
    for (int i= 0; i < num_columns; i++) {
        for (int j= 0; j < num_elts_list[i]; j++) {
            result(z++, 0)= column_list[i](j, 0);
        }
    }

    return result;
}

void insertColumn(MatrixXd &result, const MatrixXd &column, int col_num) {
    int n= (int) column.rows();
    for (int i= 0; i < n; i++) {
        result(i, col_num)= column(i, 0);
    }
}

// Assumes length of A is a perfect square
MatrixXd convertColToMatrix(const MatrixXd &column) {
    int n= (int) sqrt(column.rows());
    MatrixXd result(n, n);

    for (int i= 0; i < column.rows(); i++) {
        result(i/n, i%n) = column(i, 0);
    }

    return result;
}

MatrixXd convertMatrixToCol(const MatrixXd &matrix) {
    int n= (int) matrix.rows();
    MatrixXd result(n*n, 1);

    int z= 0;
    for (int i= 0; i < n; i++) {
        for (int j= 0; j < n; j++) {
            result(z++, 0)= matrix(i, j);
        }
    }

    return result;
}

// Assumes filter dimensions are 3x3 and assumes A is larger than a 2x2
MatrixXd convolve(const MatrixXd &A, const MatrixXd &filter) {
    int a= (int) A.rows();
    int b= a - 2;
    MatrixXd result(b, b);

    // Coordinates of resulting matrix
    int x= 0, y= 0;

    // Fill the resulting matrix
    while (y < b) {

        double sum= 0;

        // Fill in the patch
        for (int c= -1; c <= 1; c++) {
            for (int d= -1; d <= 1; d++) {
                sum += A(x+c+1, y+d+1)*filter(c+1, d+1);
            }
        }

        result(x,y)= sum/9;

        if (++x == b) {
            x= 0;
            y++;
        }
    }

    return result;
}

void conv_print_batch(const MatrixXd &C, const MatrixXd &Y, int size) {
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
        for (int j = 0; j < CONVOLUTION_OUTPUT_SIZE; j++) {
            if (j != 0 && j % POOLING_OUTPUT_SIZE == 0) {
                cout << "\n";
            }
            if (C(j, i) < 0) {
                cout << "@.@"; // Represents dark pixel
            } else {
                cout << " . "; // Represents light pixel
            }
        }
        cout << "\n";
    }
}

MatrixXd conv_get_image_batch(const int offsets[], int index, int size, const string &path) {
    // Create X Matrix of dimension CONVOLUTION_OUTPUT_SIZE x size to represent input layer
    MatrixXd X = MatrixXd::Zero(CONVOLUTION_OUTPUT_SIZE, size);

    // Open file
    ifstream images_file(path, ios::in | ios::binary);

    if (images_file.is_open()) {
        // Extract size number of random images
        for (int i = 0; i < size; i++) {
            images_file.seekg(CONVOLUTION_OUTPUT_SIZE * offsets[index + i] * sizeof(double));
            for (int j= 0; j < CONVOLUTION_OUTPUT_SIZE; j++) {
                double temp = 0;
                images_file.read((char *) &temp, sizeof(double));

                X(j % 784, i) = temp;
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
