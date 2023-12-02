# Importing necessary libraries and modules
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from joblib import dump, load
import numpy as np
import pandas as pd
import os

# Assuming these utility functions are defined in utils.py
from utils import read_digits, train_test_dev_split, predict_and_eval

# Define constants
NUM_RUNS = 1
TEST_SIZES = [0.2]
DEV_SIZES = [0.2]
ROLL_NUMBER = "123456"  # Replace with your roll number

# Preprocessing function with unit normalization
def preprocess_data(X):
    return normalize(X, norm='l2', axis=1, copy=True)

# Reading and splitting the dataset
X, y = read_digits()

# Hyperparameters for SVM, Decision Tree, and Logistic Regression
classifier_param_dict = {
    'svm': [{'gamma': g, 'C': c} for g in [0.0001, 0.0005, 0.001, 0.01, 0.1, 1] for c in [0.1, 1, 10, 100, 1000]],
    'tree': [{'max_depth': d} for d in [5, 10, 15, 20, 50, 100]],
    'lr': [{'solver': solver} for solver in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']]
}

for cur_run_i in range(NUM_RUNS):
    for test_size in TEST_SIZES:
        for dev_size in DEV_SIZES:
            train_size = 1 - test_size - dev_size
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)

            for model_type, hparams in classifier_param_dict.items():
                for param in hparams:
                    if model_type == 'svm':
                        model = svm.SVC(**param)
                    elif model_type == 'tree':
                        model = DecisionTreeClassifier(**param)  # Assuming DecisionTreeClassifier is imported
                    elif model_type == 'lr':
                        model = LogisticRegression(**param)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                        mean_score, std_score = np.mean(cv_scores), np.std(cv_scores)
                        print(f"LR with {param['solver']} - Mean CV Score: {mean_score}, Std: {std_score}")

                    model.fit(X_train, y_train)
                    test_acc, test_f1, predicted_y = predict_and_eval(model, X_test, y_test)
                    print(f"{model_type} with {param} - Test Accuracy: {test_acc}, Test F1: {test_f1}")

                    # Save the model
                    filename = f"{ROLL_NUMBER}_{model_type}_{param['solver'] if 'solver' in param else param}.joblib"
                    dump(model, filename)





