# Digit-Recognition
This Digit Recognition Program is a C++ application that implements a neural network from scratch using Eigen3, a powerful linear algebra library. It utilizes deep learning techniques to recognize handwritten digits given from the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset. The program takes inputs 28x28 pixel images of a handwritten digit and predicts the corresponding numerical value.

## Installation and Usage
To use this program, first clone the repository. Then, simply compile and run `network.cpp` and `run-tests.cpp` with the following compilation options:
`./src/functions.cpp -I "./lib/Eigen3" -O3 -DNDEBUG`

Before compiling you can optionally adjust the training parameters located in `functions.h` by changing the value of the `#define`'s. Here's a list of the following parameters:
| Parameters | Description |
| --- | --- |
| TRAIN_LABELS_FILE_PATH | Path of the file containing the training image labels |
| TRAIN_IMAGES_FILE_PATH | Path of the file containing the training image data |
| TEST_LABELS_FILE_PATH | Path of the file containing the test image labels  |
| TEST_IMAGES_FILE_PATH | Path of the file containing the test image data |
| WEIGHTS_AND_BIASES_FILE_PATH | Path of the file containing the values weights and biases resulting from training |
| LABEL_START | Offset where label data begins in the file |
| IMAGE_START | Offset where image data begins in the file |
| BATCH_SIZE | Size of the mini-batches used for training |
| NUM_TRAIN_IMAGES | Number of training images |
| NUM_TEST_IMAGES | Number of testing images |
| LEARNING_RATE | Learning rate of the neural network |
| L1_SIZE | Number of neurons in hidden layer |
| NUM_EPOCHS | Number of training epochs |
| ACTIVATION_FUNCTION | Denotes which activation function to use |
| PRINT_IMAGE_AND_LABEL | Denotes whether to print the image and label to console |

Also note, after network.exe is done running, it saves the values of its weights and biases into the binary file `wandb.bin`. `run-tests2.exe` then reads these weights and biases from `wandb.bin` to test the accuracy of the neural network on the test set.
## Algorithm
This program uses a neural network containing a single hidden layer. The program includes a pre-trained model that has been trained on 60,000 images from the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset. It utilizes a learning rate of 0.01, has 350 hidden layer neurons, and undergoes 1000 training epochs. After training, the model achieves an accuracy of around 88%. The weights and biases of the model are saved into `wandb.bin` for future use.

## Resources
#### MNIST Dataset
The program was trained using the [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset, which is a popular benchmark dataset for digit recognition tasks. The [MNIST](http://yann.lecun.com/exdb/mnist/index.html) dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is a grayscale image of size 28x28 pixels.

#### Eigen
This program uses [Eigen](https://gitlab.com/libeigen/eigen), a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.

## Side word
I created this project for fun, and it's my first attempt at building a machine learning program! I used C++ instead of python for its speed/efficiency, and because I wanted to implement everything from scratch. I know there are a ton of improvements I could make and things I could learn, so if you have any suggestions/comments feel free to contact me!































