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


# returns the first ten instances of every class, in a vector such that
# the first ten elements have label 0, the ten elements after that have
# label 1, ..., the last ten elements have label 9
def find_first_ten_instances(images, labels):
    found_all = False
    n_found = [0] * 10
    images_reordered = [0] * 100
    i = 0
    while not found_all:
        if n_found[labels[i]] < 10:
            label = labels[i]
            images_reordered[label*10+n_found[label]] = images[i]
            n_found[label] += 1
        if all(n_found[j] == 10 for j in range(10)):
            found_all = True
        i += 1
    return images_reordered


def vector_to_matrix(vector):
    matrix = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            matrix[i][j] = vector[i * 28 + j]
    return matrix


# show plot with 10 examples of each class
def plotData(images, labels):
    labels_string = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    images_reordered = find_first_ten_instances(images, labels)
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(0, 10):
        for j in range(0, 10):
            pic = vector_to_matrix(images_reordered[i*10+j])
            ax[i, j].imshow(pic, cmap='gist_gray', origin='upper')
            ax[i, j].axes.yaxis.set_visible(False)
            if j == 0:
                ax[i, j].axes.xaxis.set_ticks([])
                ax[i, j].set_xlabel('label: ' + labels_string[i], loc='left')
            else:
                ax[i, j].axes.xaxis.set_visible(False)
    plt.tight_layout(pad=0)
    plt.show()
    fig.savefig('data_plot.png')
