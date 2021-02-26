import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, AvgPool2d, Module, BatchNorm2d, Dropout, \
    Sigmoid, LeakyReLU, Tanh


class CNN(Module):

    def __init__(self, cnn_type, dropout_rate=None):
        super(CNN, self).__init__()
        if cnn_type == 'standard':
            self.cnn_layers = Sequential(
                # 2D convolution layer
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2),
                # 2D convolution layer
                Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'dropout':
            self.cnn_layers = Sequential(
                # 2D convolution layer
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                Dropout(dropout_rate),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2),
                # 2D convolution layer
                Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                Dropout(dropout_rate),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'no-batch':
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
                MaxPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'one-layer':
            self.cnn_layers = Sequential(
                # 2D convolution layer
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'sigmoid':
            self.cnn_layers = Sequential(
                # 2D convolution layer
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                Sigmoid(),
                MaxPool2d(kernel_size=2, stride=2),
                # 2D convolution layer
                Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                Sigmoid(),
                MaxPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'leaky-relu':
            self.cnn_layers = Sequential(
                # 2D convolution layer
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                LeakyReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2),
                # 2D convolution layer
                Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(4),
                LeakyReLU(inplace=True),
                MaxPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'lenet5relu':
            self.cnn_layers = Sequential(
                Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                BatchNorm2d(6),
                ReLU(inplace=True),
                AvgPool2d(kernel_size=2, stride=2, padding=0),
                Conv2d(6, 16, kernel_size=5, stride=1, padding=1),
                BatchNorm2d(16),
                ReLU(inplace=True),
                AvgPool2d(kernel_size=2, stride=2)
            )

        if cnn_type == 'lenet5tanh':
            self.cnn_layers = Sequential(
                Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                BatchNorm2d(6),
                Tanh(),
                AvgPool2d(kernel_size=2, stride=2, padding=0),
                Conv2d(6, 16, kernel_size=5, stride=1, padding=1),
                BatchNorm2d(16),
                Tanh(),
                AvgPool2d(kernel_size=2, stride=2)
            )

        # put random tensor through convolutional layers to determine dimensions for linear layers
        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        x = self.cnn_layers(x)
        self.input_size_linear = x.shape[1] * x.shape[2] * x.shape[3]

        if cnn_type == 'lenet5relu' or cnn_type == 'lenet5tanh':
            if cnn_type == 'lenet5relu':
                self.linear_layers = Sequential(
                    Linear(self.input_size_linear, 140),
                    ReLU(inplace=True),
                    Linear(140, 84),
                    ReLU(inplace=True),
                    Linear(84, 10)
                )
            if cnn_type == 'lenet5tanh':
                self.linear_layers = Sequential(
                    Linear(self.input_size_linear, 140),
                    Tanh(),
                    Linear(140, 84),
                    Tanh(),
                    Linear(84, 10)
                )
        else:
            self.linear_layers = Sequential(
                Linear(self.input_size_linear, 10)
            )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, self.input_size_linear)
        x = self.linear_layers(x)
        return x
