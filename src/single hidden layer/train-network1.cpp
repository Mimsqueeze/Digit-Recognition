#include <fstream>
#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "../functions.h"
#include "network1.h"

using namespace std;
using Eigen::MatrixXd;

// Initialize weights and biases to a random value between -0.5 and 0.5
MatrixXd W1 = MatrixXd::Random(L1_SIZE, 784)/2;
MatrixXd B1 = MatrixXd::Random(L1_SIZE, 1)/2;
MatrixXd W2 = MatrixXd::Random(10, L1_SIZE)/2;
MatrixXd B2 = MatrixXd::Random(10, 1)/2;

void save_weights_and_biases() {
    cout << "Saving weights and biases to file...\n";
    streamoff write_position = 0;
    write_position = save(W1, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    write_position = save(B1, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    write_position = save(W2, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
    save(B2, write_position, WEIGHTS_AND_BIASES_FILE_PATH);
}

void signal_callback_handler(int signum) {
    // Optionally save weights and biases to file
    if (SAVE_WEIGHTS_AND_BIASES) {
        save_weights_and_biases();
    }

    exit(signum);
}

int main() {
    // Register signal handler
    signal(SIGINT, signal_callback_handler);

    // Randomize the starting seed
    srand((unsigned int) time(nullptr));

    // For each epoch, perform gradient descent and update weights and biases
    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        // Get start time
        auto start = chrono::high_resolution_clock::now();

        // Store number of correct predictions
        int count= gradient_descent(&W1, &B1, &W2, &B2, LEARNING_RATE, epoch);

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

    cout << "Finished training!\n";

    // Optionally save weights and biases to file
    if (SAVE_WEIGHTS_AND_BIASES) {
        save_weights_and_biases()
    }

    return 0;
}