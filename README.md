# DL_ImageCNN

A convolutional neural network to classify the images from the Mnist Fashion dataset. The program will first optimize a hyperparameter, which one depends on the model that is ran.
It will then perform 10 runs of training the network on the training data and make predictions on the testing data, after which the average accuracy and standard deviation will be returned.


## Setup

Requires: 

Python 3.8.5
numpy 1.19.5
torch 1.7.1
matplotlib 3.3.3
tqdm 4.56.2
## Running the code

The code with the standard model can be ran with the command:
python3 launch.py optimizer adam

The other versions of the program can be ran with the commands:
For the model with weight decay:
python3 launch.py

For the model with the rmsprop optimizer:
python3 launch.py optimizer rmsprop

For the model with only one layer:
python3 launch.py type one-layer

For the model with a sigmoid activation layer:
python3 launch.py type sigmoid

for the leaky-ReLU activation layer:
python3 launch.py type leaky-relu

for the model without batchnormalization:
python3 launch.py type no-batch

For the model with a dropout layer:
python3 launch.py dropout

For the model with the LeNet5 architecture with a relu activation layer:
python3 launch.py type lenet5relu

For the model with the LeNet5 architecture with a tanh activation layer:
python3 launch.py type lenet5tanh

To optimize the learning rate of a specific architecture or a activation function(e.g. the LeNet5 architecture with a relu activation layer):
python3 launch.py learning-rate lenet5relu




