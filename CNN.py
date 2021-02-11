import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from tqdm import tqdm
import numpy as np


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def train_model(train_x, train_y, epochs=20):
    model = CNN()
    # model = model.float()
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss()

    loss_list = []
    for epoch in tqdm(range(0, epochs)):
        for i in range(len(train_x)):
            outputs = model.forward(train_x[i])
            train_y = train_y.long()
            loss = criterion(outputs, train_y[i])
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, np.mean(loss_list)


def eval_cnn(test_x, test_y, model):
    output = model.forward(test_x)
    total = test_y.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = 0
    for i in range(total):
        if predicted[i] == test_y[i]:
            correct += 1
    return correct / total


def train_and_test_model(train_x, train_y, test_x, test_y, n=5):
    train_acc = []
    test_acc = []
    for i in tqdm(range(n)):
        model, loss = train_model(train_x, train_y)
        train_acc.append(1-loss)
        acc = eval_cnn(test_x, test_y, model)
        test_acc.append(acc)
    return np.mean(train_acc), np.mean(test_acc)
