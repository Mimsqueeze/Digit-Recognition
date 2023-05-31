#include <iostream>
#include "network.h"
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

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

int get_num_correct(const MatrixXd &P, const MatrixXd &Y) {
    int correct= 0;
    for (int i= 0; i < BATCH_SIZE; i++) {
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
