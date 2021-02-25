import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop
from tqdm import tqdm
import numpy as np

from CNN import CNN

def train_model(train_x, train_y, cnn_type, epochs=50, learning_rate=0.01, weight_decay=0.01, batch_size=None, optimizer = None):
    model = CNN(cnn_type).float()
    if optimizer:
        optimizer = select_optimizer(optimizer, model.parameters())
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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


def train_and_test_model(train_x, train_y, test_x, test_y, cnn_type,
                         n_runs=5, epochs=50, learning_rate=0.01, weight_decay=0.01, batch_size=None, optimizer = None):
    train_acc = 0
    test_acc = 0
    for i in range(n_runs):
        if n_runs != 1:
            print('run', i+1, '/', n_runs)
        model = train_model(train_x, train_y, cnn_type,
                                  epochs=epochs,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay,
                                  batch_size=batch_size,
                                  optimizer = optimizer)
        acc = evaluate_model(train_x, train_y, model)
        train_acc = (train_acc + acc)
        acc = evaluate_model(test_x, test_y, model)
        test_acc = (test_acc + acc)
    return train_acc/n_runs, test_acc/n_runs


def set_hyperparameter(hyperparameter):
    if hyperparameter == 'weight decay':
        m_name = 'weight decay'
        start = 0
        m_range = np.array([0, 0.1, 0.001, 0.001, 0.0001, 0.00001])

    if hyperparameter == 'epochs':
        m_name = 'epochs'
        start = 25
        stop = 200
        step = 25
        m_range = np.arange(start, stop, step)
    return m_name, m_range, start


def select_optimizer(optimizer, parameters):
    if optimizer == 'adam':
        optimizer = Adam(parameters)
    if optimizer == 'rmsprop':
        optimizer = RMSprop(parameters)
    return optimizer

    
def choose_train_and_test_model(train_x, train_y, valid_x, valid_y, m, cnn_type='standard', hyperparameter = 'epochs', optimizer = None, n_runs = 1, epochs = 200):
    if hyperparameter == 'weight decay':
        acc_train, acc_valid = train_and_test_model(train_x, train_y, valid_x, valid_y, cnn_type, n_runs = n_runs, epochs = epochs, weight_decay = m)
    if hyperparameter == 'epochs':
        acc_train, acc_valid = train_and_test_model(train_x, train_y, valid_x, valid_y, cnn_type, n_runs = n_runs, epochs = m, optimizer = optimizer)
    return acc_train, acc_valid

def cross_validation(images, labels, k, hyperparameter, cnn_type = 'standard', optimizer = None):
    # setup the k-fold split
    folds_x = list(torch.chunk(images, k))
    folds_y = list(torch.chunk(labels, k))

    # range of the parameter m that is optimized
    # this parameter is also set equal to m in the train_and_test_model call below
    m_name, m_range, start = set_hyperparameter(hyperparameter)

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
            acc_train, acc_valid = choose_train_and_test_model(train_x, train_y, valid_x, valid_y, m, cnn_type, hyperparameter, optimizer)
                        
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