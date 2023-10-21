"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm

from utils import preprocess_data, split_data, train_model, read_digits, predict_and_eval, train_test_dev_split, get_hyperparameter_combinations, tune_hparams
from joblib import dump, load
import pandas as pd

num_runs  = 5
# 1. Get the dataset
X, y = read_digits()

# 2. Hyperparameter combinations
classifier_param_dict = {}
# 2.1. SVM
gamma_list = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
C_list = [0.1, 1, 10, 100, 1000]
h_params={}
h_params['gamma'] = gamma_list
h_params['C'] = C_list
h_params_combinations = get_hyperparameter_combinations(h_params)
classifier_param_dict['svm'] = h_params_combinations

# 2.2 Decision Tree
max_depth_list = [5, 10, 15, 20, 50, 100]
h_params_tree = {}
h_params_tree['max_depth'] = max_depth_list
h_params_trees_combinations = get_hyperparameter_combinations(h_params_tree)
classifier_param_dict['tree'] = h_params_trees_combinations


results = []
test_sizes =  [0.2]
dev_sizes  =  [0.2]
for cur_run_i in range(num_runs):
    
    for test_size in test_sizes:
        for dev_size in dev_sizes:
            train_size = 1- test_size - dev_size
            # 3. Data splitting -- to create train and test sets                
            X_train, X_test, X_dev, y_train, y_test, y_dev = train_test_dev_split(X, y, test_size=test_size, dev_size=dev_size)
            # 4. Data preprocessing
            X_train = preprocess_data(X_train)
            X_test = preprocess_data(X_test)
            X_dev = preprocess_data(X_dev)


            for model_type in classifier_param_dict:
                current_hparams = classifier_param_dict[model_type]
                best_hparams, best_model_path, best_accuracy  = tune_hparams(X_train, y_train, X_dev, 
                y_dev, current_hparams, model_type)        
            
                # loading of model         
                best_model = load(best_model_path) 

                test_acc = predict_and_eval(best_model, X_test, y_test)
                train_acc = predict_and_eval(best_model, X_train, y_train)
                dev_acc = best_accuracy

                print("{}\ttest_size={:.2f} dev_size={:.2f} train_size={:.2f} train_acc={:.2f} dev_acc={:.2f} test_acc={:.2f}".format(model_type, test_size, dev_size, train_size, train_acc, dev_acc, test_acc))
                cur_run_results = {'model_type': model_type, 'run_index': cur_run_i, 'train_acc' : train_acc, 'dev_acc': dev_acc, 'test_acc': test_acc}
                results.append(cur_run_results)

print(pd.DataFrame(results).groupby('model_type').describe().T)
#changes made for quiz 2

# 9. Evaluate Model Accuracy:
# Calculate the accuracy of both models on the test data
production_predictions = production_model.predict(X_test)
candidate_predictions = candidate_model.predict(X_test)
production_accuracy = accuracy_score(y_test, production_predictions)
candidate_accuracy = accuracy_score(y_test, candidate_predictions)

# 10. Calculate Confusion Matrices:
# Calculate the confusion matrix for both models
production_confusion_matrix = confusion_matrix(y_test, production_predictions)
candidate_confusion_matrix = confusion_matrix(y_test, candidate_predictions)

# 11. Calculate Specific Confusion Matrix:
# To find out how many samples are predicted correctly in production but not in the candidate,
# you can create a specific confusion matrix comparing the two models
common_samples = (production_predictions == y_test) & (candidate_predictions != y_test)
specific_confusion_matrix = confusion_matrix(y_test[common_samples], production_predictions[common_samples])

# 12. Calculate Macro-average F1 Scores (Bonus):
# To calculate the macro-average F1 scores, you can use the f1_score function with the average='macro' parameter
production_f1 = f1_score(y_test, production_predictions, average='macro')
candidate_f1 = f1_score(y_test, candidate_predictions, average='macro')

# 13. Display or Log the Results:
# You can print or log the results to see the accuracy, confusion matrices, and F1 scores of both models
print("Production Model's Accuracy:", production_accuracy)
print("Candidate Model's Accuracy:", candidate_accuracy)
print("Production Model's Confusion Matrix:")
print(production_confusion_matrix)
print("Candidate Model's Confusion Matrix:")
print(candidate_confusion_matrix)
print("Specific Confusion Matrix (Production Model's Correct Predictions, Candidate Model's Incorrect Predictions):")
print(specific_confusion_matrix)
print("Production Model's Macro-average F1 Score:", production_f1)
print("Candidate Model's Macro-average F1 Score:", candidate_f1)





                

