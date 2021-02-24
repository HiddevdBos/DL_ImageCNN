import sys
import torch
from readData import get_data
from runModel import choose_train_and_test_model, cross_validation
from plots import plotTrainTestError

def show_error():
	print("Please pass the required arguments")
	sys.exit()

if __name__ == '__main__':

	train_images, train_labels = get_data('fashion-mnist_train.csv')
	test_images, test_labels = get_data('fashion-mnist_test.csv')

	if len(sys.argv) < 2:
		# ---- cross-validation ----
		best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, k=5, hyperparameter = 'weight decay')
		plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
		print(f'optimal value for {m_name}: {best_m}')

		# ---- testing ----
		train_acc, test_acc = choose_train_and_test_model(train_images, train_labels, test_images, test_labels, best_m, n_runs=5, epochs=200, hyperparameter = 'weight decay')
		print('average training accuracy:', train_acc)
		print('average testing accuracy:', test_acc)



  #------------------Optimizer--------------------#
	elif(sys.argv[1] == 'optimizer'):
		if len(sys.argv) < 3:
			show_error()
		# ---- cross-validation ----
		best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, optimizer = sys.argv[2], k=5, hyperparameter = 'epochs')
		plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
		print(f'optimal value for {m_name}: {best_m}')

		# ---- testing ----
		train_acc, test_acc = choose_train_and_test_model(train_images, train_labels, test_images, test_labels, best_m, optimizer = sys.argv[2], n_runs=5, hyperparameter = 'epochs')
		print('average training accuracy:', train_acc)
		print('average testing accuracy:', test_acc)
