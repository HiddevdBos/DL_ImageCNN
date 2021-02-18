import numpy as np
import matplotlib.pyplot as plt


def plotTrainTestError(train_acc, test_acc, m_name, save_fig=False, x_values=[]):
    train_acc[:] = [1 - x for x in train_acc]
    test_acc[:] = [1 - x for x in test_acc]
    if not x_values:
        plt.plot(train_acc)
        plt.plot(test_acc)
    else:
        plt.plot(x_values, train_acc)
        plt.plot(x_values, test_acc)
    plt.gca()
    plt.xlabel(m_name)
    plt.ylabel('Error rate')
    plt.legend(['Training', 'Validation'], loc=1)
    if save_fig:
        plt.savefig(f'crossval_plot_{m_name}.png')
    plt.show()


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