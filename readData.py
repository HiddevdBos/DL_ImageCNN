import zipfile
import numpy as np
import torch
from plots import plotData


# load data and labels from zipfile, and store in numpy arrays
def load_data(path, plot=False):
    with zipfile.ZipFile('data.zip') as data:
        data_file = data.open(path)
    images = []
    labels = []
    for idx, line in enumerate(data_file):
        if idx == 0:
            continue
        line = str(line)[:-3]
        image = line.split(',')
        labels.append(int(image[0][2:]))
        image = list(map(int, image[1:]))
        images.append(image)
    if plot:
        plotData(images, labels)
    return np.array(images).reshape(len(images), 1, 28, 28), np.array(labels)


# load data, change data to floats and get data to gpu (if cuda available)
def get_data(path, plot=False):
    data, labels = load_data(path, plot)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels
