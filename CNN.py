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
            # 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
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


def train_model(train_x, train_y, epochs=20, learning_rate=0.01, weight_decay=0.01, batch_size=None):
    model = CNN()
    model = model.float()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    # default batch size is entire dataset
    if not batch_size or batch_size > train_x.shape[0]:
        batch_size = train_x.shape[0]
    num_batches = int(train_x.shape[0] / batch_size)

    train_x = train_x.reshape(-1, batch_size, 1, 28, 28)
    train_y = train_y.reshape(-1, batch_size)

    loss_list = []
    for epoch in tqdm(range(0, epochs), desc='epochs'):
        for i in range(num_batches):
            # forward step
            outputs = model.forward(train_x[i])
            train_y = train_y.long()
            loss = criterion(outputs, train_y[i])
            loss_list.append(loss.item())

            # backwards step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # return the model and the average training loss
    return model, np.mean(loss_list)


def evaluate_model(test_x, test_y, model):
    output = model.forward(test_x)
    total = test_y.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = 0
    for i in range(total):
        if predicted[i] == test_y[i]:
            correct += 1
    # return testing accuracy
    return correct / total


def train_and_test_model(train_x, train_y, test_x, test_y,
                         n_runs=5, epochs=20, learning_rate=0.01, weight_decay=0.01, batch_size=None):
    train_acc = []
    test_acc = []
    for i in range(n_runs):
        print('run', i+1, '/', n_runs)
        model, loss = train_model(train_x, train_y,
                                  epochs=epochs,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay,
                                  batch_size=batch_size)
        acc = evaluate_model(train_x, train_y, model)
        train_acc.append(acc)
        acc = evaluate_model(test_x, test_y, model)
        test_acc.append(acc)
    return np.mean(train_acc), np.mean(test_acc)


def cross_validation(images, labels, k):
    # setup the k-fold split
    folds_x = list(torch.chunk(images, k))
    folds_y = list(torch.chunk(labels, k))

    acc_train_m = list()
    acc_valid_m = list()
    m_list = list()

    # range of the parameter m that is optimized
    # this parameter is also set equal to m in the train_and_test_model call below
    m_name = 'weight decay'
    start = 0
    stop = 0.02
    step = 0.01

    best_m = 0
    best_m_acc = 0
    m_range = np.arange(start, stop, step)
    print(f'training and evaluating', k * len(m_range), 'models')

    for m in tqdm(m_range, desc='m values'):
        acc_train_list = list()
        acc_valid_list = list()

        # train a new model for each fold
        for fold in range(0, k):
            train_x = folds_x.copy()
            train_y = folds_y.copy()
            valid_x = train_x.pop(fold)
            valid_y = train_y.pop(fold)
            train_x = torch.cat(train_x)
            train_y = torch.cat(train_y)

            acc_train, acc_valid = train_and_test_model(train_x, train_y, valid_x, valid_y, epochs=2,
                                                        weight_decay=m, batch_size=50, n_runs=1)
            acc_train_list.append(acc_train)
            acc_valid_list.append(acc_valid)
        mean_train_acc = round(np.mean(acc_train_list), 4)
        mean_valid_acc = round(np.mean(acc_valid_list), 4)
        acc_train_m.append(mean_train_acc)
        acc_valid_m.append(mean_valid_acc)

        # saving the optimal value for m
        if mean_valid_acc > best_m_acc:
            best_m_acc = mean_valid_acc
            best_m = m
        m_list.append(round(m, 4))

    # return the optimal value for m, the training accuracies, validation accuracies,
    # and for the plot the range of m and the name of m
    return best_m, acc_train_m, acc_valid_m, m_list, m_name
