import zipfile
import numpy as np
import matplotlib.pyplot as plt
import torch


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
        plotDigits(data, labels)
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
    labels = one_hot_encode_labels(labels)
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)
    data = data.float()
    if torch.cuda.is_available():
        data = data.cuda()
        labels = labels.cuda()
    return data, labels


def find_first_ten_instances(images, labels):
    found = False
    found_length = [0] * 10
    matrix = [[0]*10]*10
    i = 0
    while not found:
        if found_length[labels[i]] < 10:
            matrix[labels[i]][found_length[labels[i]]-1] = images[i]
            found_length[labels[i]] += 1
        if all(found_length[j] == 10 for j in range(10)):
            found = True
        i += 1
    return matrix


def vector_to_matrix(vector):
    matrix = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            matrix[i][j] = vector[i*28+j]
    return matrix


# show plot with 10 examples of each digit
def plotDigits(images, labels):
    matrix = find_first_ten_instances(images, labels)
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(0, 10):
        for j in range(0, 10):
            pic = vector_to_matrix(matrix[i][j])
            ax[i, j].pcolor(pic, cmap='gist_gray')
            # ax[i, j].imshow(pic, cmap='gist_gray')
            ax[i, j].axes.xaxis.set_visible(False)
            ax[i, j].axes.yaxis.set_visible(False)
    plt.show()
    fig.savefig('digits.png')