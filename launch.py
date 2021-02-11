from readData import get_data
from CNN import train_and_test_model

train_images, train_labels = get_data('fashion-mnist_train.csv', plot=False)
print(len(train_images))
print(train_images.size())
print(train_labels.size())
test_images, test_labels = get_data('fashion-mnist_test.csv')
print(test_images.size())
print(test_labels.size())

train_acc, test_acc = train_and_test_model(train_images, train_labels, test_images, test_labels)
print(train_acc, test_acc)
