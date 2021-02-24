import zipfile
import numpy as np
import torch

# load data and labels from zipfile, and store in numpy arrays
def load_data(path):
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
    return np.array(data).reshape(len(data), 1, 28, 28), np.array(labels)

# load data, change data to floats and get data to gpu (if cuda available)
def get_data(path):
    data, labels = load_data(path)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels