import sys
from readData import get_data
from runModel import choose_train_and_test_model, cross_validation, train_and_test_model
from plots import plotTrainTestError


def show_error():
    print("Please pass the required arguments")
    sys.exit()


if __name__ == '__main__':

    train_images, train_labels = get_data('fashion-mnist_train.csv')
    test_images, test_labels = get_data('fashion-mnist_test.csv')

	# run model with weight decay
    if len(sys.argv) < 2:
        # ---- cross-validation ----
        best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, k=5,
                                                                        cnn_type='no-batch',
                                                                        hyperparameter='weight decay')
        plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
        print(f'optimal value for {m_name}: {best_m}')

        # ---- testing ----
        acc_train, acc_train_sd, acc_test, acc_test_sd = choose_train_and_test_model(train_images, train_labels,
                                                                                     test_images, test_labels, best_m,
                                                                                     n_runs=10, cnn_type='no-batch',
                                                                                     epochs=200,
                                                                                     hyperparameter='weight decay')
        print('test = ', sys.argv)
        print('average training accuracy:', acc_train)
        print('standard deviation:', acc_train_sd)
        print('average testing accuracy:', acc_test)
        print('standard deviation:', acc_test_sd)

    
    # run the model with a certain optimizer 
    else:
        if sys.argv[1] == 'optimizer':
            if len(sys.argv) < 3:
                show_error()
            # ---- cross-validation ----
            best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels,
                                                                            cnn_type='standard',
                                                                            optimizer=sys.argv[2], k=5,
                                                                            hyperparameter='epochs')
            plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
            print(f'optimal value for {m_name}: {best_m}')

            # ---- testing ----
            acc_train, acc_train_sd, acc_test, acc_test_sd = choose_train_and_test_model(train_images, train_labels,
                                                                                         test_images, test_labels,
                                                                                         cnn_type='standard', m=best_m,
                                                                                         n_runs=10,
                                                                                         optimizer=sys.argv[2],
                                                                                         hyperparameter='epochs')
            print('test = ', sys.argv)
            print('average training accuracy:', acc_train)
            print('standard deviation:', acc_train_sd)
            print('average testing accuracy:', acc_test)
            print('standard deviation:', acc_test_sd)

        # run the model with a certain architecture
        if sys.argv[1] == 'type':
            if len(sys.argv) < 3:
                show_error()
            # ---- cross-validation ----
            best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, k=5,
                                                                            cnn_type=sys.argv[2],
                                                                            hyperparameter='epochs')
            plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
            print(f'optimal value for {m_name}: {best_m}')

            # ---- testing ----
            acc_train, acc_train_sd, acc_test, acc_test_sd = choose_train_and_test_model(train_images, train_labels,
                                                                                         test_images, test_labels,
                                                                                         best_m, n_runs=10,
                                                                                         cnn_type=sys.argv[2],
                                                                                         hyperparameter='epochs')
            print('test = ', sys.argv)
            print('average training accuracy:', acc_train)
            print('standard deviation:', acc_train_sd)
            print('average testing accuracy:', acc_test)
            print('standard deviation:', acc_test_sd)

        # optimize the model for the learning rate
        if sys.argv[1] == 'learning_rate':
            # ---- cross-validation ----
            best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, k=5,
                                                                            cnn_type=sys.argv[2],
                                                                            hyperparameter='learning rate')
            plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
            print(f'optimal value for {m_name}: {best_m}')

            # ---- testing ----
            acc_train, acc_train_sd, acc_test, acc_test_sd = choose_train_and_test_model(train_images, train_labels,
                                                                                         test_images, test_labels,
                                                                                         best_m,
                                                                                         n_runs=10,
                                                                                         cnn_type=sys.argv[2],
                                                                                         epochs=200,
                                                                                         hyperparameter='learning rate')
            print('test = ', sys.argv)
            print('average training accuracy:', acc_train)
            print('standard deviation:', acc_train_sd)
            print('average testing accuracy:', acc_test)
            print('standard deviation:', acc_test_sd)

        # train a model with a dropout layer and optimize amount of knowledge that is dropped out
        if sys.argv[1] == 'dropout':
            # ---- cross-validation ----
            best_m, acc_train, acc_valid, m_list, m_name = cross_validation(train_images, train_labels, k=5,
                                                                            cnn_type='dropout',
                                                                            hyperparameter='dropout rate')
            plotTrainTestError(acc_train, acc_valid, m_name, x_values=m_list)
            print(f'optimal value for {m_name}: {best_m}')

            # ---- testing ----
            acc_train, acc_train_sd, acc_test, acc_test_sd = choose_train_and_test_model(train_images, train_labels,
                                                                                         test_images, test_labels,
                                                                                         best_m, n_runs=10,
                                                                                         cnn_type='dropout',
                                                                                         hyperparameter='dropout rate')
            print('test = ', sys.argv)
            print('average training accuracy:', acc_train)
            print('standard deviation:', acc_train_sd)
            print('average testing accuracy:', acc_test)
            print('standard deviation:', acc_test_sd)
