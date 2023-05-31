#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "functions.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    srand((unsigned int) time(nullptr));

    MatrixXd W1= MatrixXd::Random(L1_SIZE, 784);
    MatrixXd B1= MatrixXd::Random(L1_SIZE, 1);
    MatrixXd W2= MatrixXd::Random(10, L1_SIZE);
    MatrixXd B2= MatrixXd::Random(10, 1);

    for (int epoch= 1; epoch <= NUM_EPOCHS; epoch++) {
        gradient_descent(&W1, &B1, &W2, &B2, LEARNING_RATE, epoch);
    }

    streamoff write_position= 0;
    write_position= save(W1, write_position);
    write_position= save(B1, write_position);
    write_position= save(W2, write_position);
    save(B2, write_position);

    return 0;
}