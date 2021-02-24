import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = Sequential(
            # 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        # put random tensor through convolutional layers to determine dimensions for linear layers
        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        x = self.cnn_layers(x)
        self.input_size_linear = x.shape[1] * x.shape[2] * x.shape[3]

        self.linear_layers = Sequential(
            Linear(self.input_size_linear, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, self.input_size_linear)
        x = self.linear_layers(x)
        return x