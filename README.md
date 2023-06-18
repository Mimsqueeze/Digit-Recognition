# Digit-Recognition
This Digit Recognition Program is a C++ application that implements different types of neural networks from scratch using Eigen3, a powerful linear algebra library. It utilizes deep learning techniques to recognize handwritten digits given from the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset. The program takes inputs 28x28 pixel images of a handwritten digit and predicts the corresponding numerical value.

## Installation and Usage
To use this program, you first need to clone to repository. Then, make sure you have C++ compiler and the make utility installed ([here's a guide! - make sure you install the full Mingw-w64 toolchain](https://code.visualstudio.com/docs/languages/cpp)). Then, simply run `make all` in the `\Digit-Recognition` directory, and then run the executables.

Source files in `./src/convolutional network` implement a convolutional neural network with a single hidden layer, `./src/single hidden layer` implement a traditional neural network with a single hidden layer, `./src/double hidden layer` implement a traditional neural network with two hidden layers. 

Run the respective `train-network.exe` executable to train the respective network, and `run-tests.exe` to test the respective network: `conv-train-network.exe`  and `conv-run-tests.exe`, `train-network1.exe` and `run-tests1.exe`, and `train-network2.exe` and `run-tests2.exe`.

After network training is finished, it saves the values of its weights and biases into the binary file `wandb.bin` in its respective folder. `run-tests.exe` then reads these weights and biases from `wandb.bin` to assess the accuracy of the neural network on the test set.

Before compiling you can optionally adjust the training parameters located in `functions.h` by changing the value of the `#define`'s. Here's a list of the following parameters:
| Parameters | Description |
| --- | --- |
| TRAIN_LABELS_FILE_PATH | Path of the file containing the training image labels |
| TRAIN_IMAGES_FILE_PATH | Path of the file containing the training image data |
| TEST_LABELS_FILE_PATH | Path of the file containing the test image labels  |
| TEST_IMAGES_FILE_PATH | Path of the file containing the test image data |
| LABEL_START | Offset where label data begins in the file |
| IMAGE_START | Offset where image data begins in the file |
| BATCH_SIZE | Size of the mini-batches used for training |
| NUM_TRAIN_IMAGES | Number of training images |
| NUM_TEST_IMAGES | Number of testing images |
| LEARNING_RATE | Learning rate of the neural network |
| NUM_EPOCHS | Number of training epochs |
| ACTIVATION_FUNCTION | Denotes which activation function to use |
| PRINT_LABELS_AND_IMAGES | Denotes whether to print the image and label to console |

You can also adjust the follwing training parameters for a network, in `conv-network.h` for the convolutional network, `network1.h` for the traditional neural network with a single hidden layer, and `network2.h` for the traditional neural network with two hidden layers:
| Parameters | Description |
| --- | --- |
| WEIGHTS_AND_BIASES_FILE_PATH | Path of the file containing the saved weights and biases |
| L1_SIZE | Number of nodes in the first hidden layer |
| L2_SIZE | Number of nodes in the second hidden layer |
| POOLING_WINDOW_SIZE | Size of the pooling window  |
| POOLING_STRIDE_SIZE | Size of the pooling stride |
| POOLING_OUTPUT_SIZE | Length of resulting image after pooling |
| CONVOLUTION_OUTPUT_SIZE | Size of the output of the convolution/pooling layer |

## Algorithm
The convolutional neural network is implemented with a singular convolutional layer utilizing vertical, horizontal, and two diagonal filters (all 3x3), a pooling layer ulilizaing max pooling (no normalization), and three fully connected layers consisting of the input layer, a single hidden layer, and the output layer. The folder `./src/convolutional network` contains a pre-trained model that has been trained with the following hyper-parameters: 60,000 training images, 0.1 learning rate, 50 hidden layer neurons, and 2500 training epochs. After training, the model achieved an accuracy of around 89%. 

The single hidden layer neural network is implemented with three fully connected layers consisting of the input layer, a single hidden layer, and the output layer. 

The double hidden layer neural network is implemented with four fully connected layers consisting of the input layer, two hidden layers, and the output layer. 

## Resources
#### MNIST Dataset
The program was trained using the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset, which is a popular benchmark dataset for digit recognition tasks. The [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is a grayscale image of size 28x28 pixels.

#### Eigen
This program uses [Eigen](https://gitlab.com/libeigen/eigen), a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

## Side word
I created this project for fun, and it's my first attempt at building a machine learning program! I used C++ instead of python for its speed/efficiency, and because I wanted to implement everything from scratch. I know there are a ton of improvements I could make and things I could learn, so if you have any suggestions/comments feel free to contact me!
