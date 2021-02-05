import zipfile

with zipfile.ZipFile('data.zip') as data:
    training_data = data.open('fashion-mnist_train.csv')
    test_data = data.open('fashion-mnist_test.csv')