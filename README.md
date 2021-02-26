# DL_ImageCNN

A convolutional neural network to classify the images from the Fashion MNIST dataset. The program will first optimize a hyperparameter, which one depends on the model that is ran.
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

The other versions of the program can be ran with the following commands:  
For the standard model with weight decay:  
python3 launch.py

For the standard model with dropout:  
python3 launch.py dropout

For the standard model without batch normalization:  
python3 launch.py type no-batch

For the standard model with the rmsprop optimizer:  
python3 launch.py optimizer rmsprop

For the model with only one layer:  
python3 launch.py type one-layer

For the standard model with sigmoid activation:  
python3 launch.py type sigmoid

For the standard model with leaky ReLU activation:  
python3 launch.py type leaky-relu

For the LeNet-5 model with ReLU activation:  
python3 launch.py type lenet5relu

For the LeNet-5 model with Tanh activation:  
python3 launch.py type lenet5tanh

To optimize the learning rate of a specific architecture or a activation function(e.g. the standard architecture with ReLU activation):  
python3 launch.py learning-rate lenet5relu




