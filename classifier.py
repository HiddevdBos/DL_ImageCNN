import zipfile
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from CNN import CNN

# load data and labels from zipfile, and store in numpy arrays
def load_data(path):             
    with zipfile.ZipFile('data.zip') as data:
        data_file = data.open(path)
    data =[]
    labels = []
    for idx, line in enumerate(data_file):
        if idx == 0:
            continue
        line = str(line)[:-3]
        image = line.split(',')
        labels.append(int(image[0][2:]))
        image = list(map(int, image[1:]))
        data.append(image)
    return np.array(data).reshape(len(data), 1, 28, 28), np.array(labels)

# binary encode the labels (niet sure of nodig)
def binary_encode_labels(labels):
    encoded_labels = []
    for label in labels:
        encoded_label = [0] * 10
        encoded_label[label] = 1
        encoded_labels.append(encoded_label)
    return np.array(encoded_labels)

# load data, encode it, change data to floats and get data to gpu (if cuda availble)
def get_data(path):
    data, labels = load_data(path)
    labels = binary_encode_labels(labels)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels


training_data, training_labels = get_data('fashion-mnist_train.csv')
test_data, test_labels = get_data('fashion-mnist_test.csv')

model = CNN()

# criterion en optimizer zijn willekeurig gekozen, geen idee of ze goed zijn
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters())

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()