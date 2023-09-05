from sklearn import metrics, svm
from utils import preprocess_data, train_model, read_digits, predict_and_eval, train_test_dev_split, tune_hparams
from itertools import product

# Define the test and dev size ranges
test_size_ranges = [0.1, 0.2, 0.3]
dev_size_ranges = [0.1, 0.2, 0.3]

# 1. Get the dataset
X, y = read_digits()

# Iterate over test_size and dev_size combinations
for test_size, dev_size in product(test_size_ranges, dev_size_ranges):
    train_size = 1.0 - test_size - dev_size

    # 3. Data splitting -- to create train and test sets
    X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)

    # 4. Data preprocessing
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    X_dev = preprocess_data(X_dev)

    # HYPER PARAMETER TUNING
    # - take all combinations of gamma and C
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]

    param_grid = {'gamma': gamma_ranges, 'C': C_ranges}

    best_hparams, best_model, best_acc = tune_hparams(X_train, y_train, X_dev, y_dev, param_grid, model_type="svm")

    print(f"test_size={test_size} dev_size={dev_size} train_size={train_size} train_acc={best_acc:.2f} dev_acc={predict_and_eval(best_model, X_dev, y_dev):.2f} test_acc={predict_and_eval(best_model, X_test, y_test):.2f}")
    print("Optimal parameters gamma:", best_hparams['gamma'], "C:", best_hparams['C'])
