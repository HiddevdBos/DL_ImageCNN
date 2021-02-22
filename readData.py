import zipfile
import numpy as np
import torch
from plots import plotData


# load data and labels from zipfile, and store in numpy arrays
def load_data(path, plot):
    with zipfile.ZipFile('data.zip') as data:
        data_file = data.open(path)
    data = []
    labels = []
    for idx, line in enumerate(data_file):
        if idx == 0:
            continue
        line = str(line)[:-3]
        image = line.split(',')
        labels.append(int(image[0][2:]))
        image = list(map(int, image[1:]))
        data.append(image)
    if plot:
        plotData(data, labels)
    return np.array(data).reshape(len(data), 1, 28, 28), np.array(labels)


# binary encode the labels (niet sure of nodig)
def one_hot_encode_labels(labels):
    encoded_labels = []
    for label in labels:
        encoded_label = [0] * 10
        encoded_label[label] = 1
        encoded_labels.append(encoded_label)
    return np.array(encoded_labels)


# load data, encode it, change data to floats and get data to gpu (if cuda available)
def get_data(path, plot=False):
    data, labels = load_data(path, plot)
    # labels = one_hot_encode_labels(labels)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels


def get_dummy_data():
    data = np.random.rand(10 ,1, 28, 28)
    labels = np.random.rand(10 ,1)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels
