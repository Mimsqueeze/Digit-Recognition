#include <fstream>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "../functions.h"
#include "network2.h"

using namespace std;
using Eigen::MatrixXd;

int main() {
    // Randomize the starting seed
    srand((unsigned int) time(nullptr));

    // Initialize weights and biases to a random value between -0.5 and 0.5
    weights_and_biases wab;
    wab.W1 = MatrixXd::Random(L1_SIZE, 784)/2;
    wab.B1 = MatrixXd::Random(L1_SIZE, 1)/2;
    wab.W2 = MatrixXd::Random(L2_SIZE, L1_SIZE)/2;
    wab.B2 = MatrixXd::Random(L2_SIZE, 1)/2;
    wab.W3 = MatrixXd::Random(10, L2_SIZE)/2;
    wab.B3 = MatrixXd::Random(10, 1)/2;

    // For each epoch, perform gradient descent and update weights and biases
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        // Get start time
        auto start = chrono::high_resolution_clock::now();

        // Store number of correct predictions
        int count= gradient_descent(wab, LEARNING_RATE, epoch);

        // Get end time
        auto end = chrono::high_resolution_clock::now();

        // Calculate duration of time passed
        double duration = (double) chrono::duration_cast<chrono::microseconds>(end - start).count()/1000000.0;

        // Calculate remaining time
        int seconds = (int) duration*(NUM_EPOCHS - epoch);
        int minutes= seconds/60;
        int hours= minutes/60;
        minutes %= 60;
        seconds %= 60;

        // Print the results of the epoch
        cout << "Epoch: " << epoch << "/" << NUM_EPOCHS << "\n";
        cout << "Accuracy: " << count << "/" << NUM_TRAIN_IMAGES << "\n";
        cout << "Time taken: " << duration << " seconds \n";
        cout << "Estimated time remaining: ";
        printf("%02d:%02d:%02d\n", hours, minutes, seconds);
        cout << "\n";
    }

    cout << "Finished training! Saving weights and biases to file...\n";

    // Optionally save weights and biases to file
    if (SAVE_WEIGHTS_AND_BIASES) {
        streamoff write_position = 0;
        write_position = save(wab.W1, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
        write_position = save(wab.B1, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
        write_position = save(wab.W2, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
        write_position = save(wab.B2, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
        write_position = save(wab.W3, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
        save(wab.B3, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    }

    return 0;
}