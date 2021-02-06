import zipfile

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
        labels.append(image[0][2:])
        image = list(map(int, image[1:]))
        data.append(image)
    return data, labels


training_data, training_labels = load_data('fashion-mnist_train.csv')
test_data, test_labels = load_data('fashion-mnist_test.csv')