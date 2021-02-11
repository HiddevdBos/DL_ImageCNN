from readData import get_data
from CNN import test_model


train_data, train_labels = get_data('fashion-mnist_train.csv', plot=False)
print(train_data.size())
print(train_labels.size())
test_data, test_labels = get_data('fashion-mnist_test.csv')
print(test_data.size())
print(test_labels.size())


