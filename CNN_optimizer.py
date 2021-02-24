import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, RMSprop, SGD
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, DataParallel
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


def train_model(train_x, train_y, optimizer, epochs=50, learning_rate=0.01, weight_decay=0.01, batch_size=None):
    model = CNN().float()
    optimizer = select_optimizer(optimizer, model.parameters())
    criterion = CrossEntropyLoss()
    if torch.cuda.is_available():
        model = DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    # default batch size is entire dataset
    if not batch_size or batch_size > train_x.shape[0]:
        batch_size = train_x.shape[0]
    num_batches = int(train_x.shape[0] / batch_size)
    train_x = train_x.reshape(-1, batch_size, 1, 28, 28)
    train_y = train_y.reshape(-1, batch_size)

    loss_list = []
    for epoch in range(0, epochs):
        for i in range(num_batches):
            # forward step
            outputs = model.forward(train_x[i])
            train_y = train_y.long()
            loss = criterion(outputs, train_y[i])
            # loss_list.append(loss.item())
            
            # backwards step
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()

    # return the model and the average training loss
    return model

def evaluate_model(test_x, test_y, model):
    output = model.forward(test_x)
    total = test_y.size(0)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == test_y).sum()
    return correct / total


def train_and_test_model(train_x, train_y, test_x, test_y, optimizer,
                         n_runs=5, epochs=50, learning_rate=0.01, weight_decay=0.01, batch_size=None):
    train_acc = 0
    test_acc = 0
    
    for i in range(n_runs):
        if n_runs != 1:
            print('run', i+1, '/', n_runs)
        model = train_model(train_x, train_y, optimizer,
                                  epochs=epochs,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay,
                                  batch_size=batch_size)
        acc = evaluate_model(train_x, train_y, model)
        train_acc = (train_acc + acc)
        acc = evaluate_model(test_x, test_y, model)
        test_acc = (test_acc + acc)
    return train_acc/n_runs, test_acc/n_runs

def select_optimizer(optimizer, parameters):
    if optimizer == 'adam':
        optimizer = Adam(parameters)
    if optimizer == 'rmsprop':
        optimizer = RMSprop(parameters)
    if optimizer == 'sgd':
        optimizer = SGD(parameters)
    return optimizer

def cross_validation(images, labels, optimizer, k):
    # setup the k-fold split
    folds_x = list(torch.chunk(images, k))
    folds_y = list(torch.chunk(labels, k))

    # range of the parameter m that is optimized
    # this parameter is also set equal to m in the train_and_test_model call below
    m_name = 'epochs'
    start = 25
    stop = 200
    step = 25

    m_range = np.arange(start, stop, step)
    print(f'training and evaluating', k * len(m_range), 'models')

    for m in tqdm(m_range, desc='m values'):
        acc_train_mean = 0
        acc_valid_mean = 0
        # train a new model for each fold
        for fold in tqdm(range(0, k), desc='folds'):
            train_x = folds_x.copy()
            train_y = folds_y.copy()
            valid_x = train_x.pop(fold)
            valid_y = train_y.pop(fold)
            train_x = torch.cat(train_x)
            train_y = torch.cat(train_y)

            # n_runs should be 1 for this            
            acc_train, acc_valid = train_and_test_model(train_x, train_y, valid_x, valid_y, optimizer, n_runs=1, epochs=m)
                        
            acc_valid_mean = (acc_valid_mean + acc_valid)
            acc_train_mean = (acc_train_mean + acc_train)

        acc_train_mean = acc_train_mean/k
        acc_valid_mean = acc_valid_mean/k

        #append accuracies to tensor
        if m == start:
            acc_train_list = acc_train_mean.unsqueeze(0)
            acc_valid_list = acc_valid_mean.unsqueeze(0)
        else:
            acc_train_list = torch.cat((acc_train_list, acc_train_mean.unsqueeze(0)), 0)
            acc_valid_list = torch.cat((acc_valid_list, acc_valid_mean.unsqueeze(0)), 0)

    #determine optimal value
    best_m = list(m_range)[torch.argmax(acc_valid_list)]

    # return the optimal value for m, the training accuracies, validation accuracies,
    # and for the plot the range of m and the name of m
    return best_m.item(), acc_train_list.cpu().tolist(), acc_valid_list.cpu().tolist(), list(m_range), m_name