from readData import get_data
from CNN import train_and_test_model, cross_validation
from plots import plotTrainTestError

train_images, train_labels = get_data('fashion-mnist_train.csv', plot=False)
# print(train_images.size())
# print(train_labels.size())
test_images, test_labels = get_data('fashion-mnist_test.csv')
# print(test_images.size())
# print(test_labels.size())

# ---- cross-validation ----
best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, k=2)
plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
print(f'optimal value for {m_name}: {best_m}')

# ---- testing ----
# train_acc, test_acc = train_and_test_model(train_images, train_labels, test_images, test_labels, n_runs=5, epochs=10)
# print('average training accuracy:', train_acc)
# print('average testing accuracy:', test_acc)
