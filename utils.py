from sklearn.model_selection import train_test_split
from sklearn import svm, datasets, metrics
from sklearn.model_selection import ParameterGrid
def tune_hparams(X_train, y_train, X_dev, y_dev, param_grid, model_type="svm"):
    best_acc_so_far = -1
    best_model = None
    best_hparams = None

    for params in ParameterGrid(param_grid):
        cur_gamma = params['gamma']
        cur_C = params['C']

        # Train the model with the current hyperparameters
        cur_model = train_model(X_train, y_train, params, model_type=model_type)

        # Get the accuracy on the dev set
        cur_accuracy = predict_and_eval(cur_model, X_dev, y_dev)

        # Select the hyperparameters that yield the best performance on the dev set
        if cur_accuracy > best_acc_so_far:
            print("New best accuracy: ", cur_accuracy)
            best_acc_so_far = cur_accuracy
            best_hparams = params
            best_model = cur_model

    return best_hparams, best_model, best_acc_so_far
def read_digits():
    digits = datasets.load_digits()
    X = digits.images
    y = digits.target
    return X, y 
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data
# Split data into 50% train and 50% test subsets
def split_data(x, y, test_size, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5,random_state=random_state
    )
    return X_train, X_test, y_train, y_test
# train the model of choice with the model prameter
def train_model(x, y, model_params, model_type="svm"):
    if model_type == "svm":
        # Create a classifier: a support vector classifier
        clf = svm.SVC
    model = clf(**model_params)
    # train the model
    model.fit(x, y)
    return model
def train_test_dev_split(X, y, test_size, dev_size):
    X_train_dev, X_test, Y_train_Dev, y_test =  train_test_split(X, y, test_size=test_size, random_state=1)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, Y_train_Dev, dev_size/(1-test_size), random_state=1)
    return X_train, X_test, X_dev, y_train, y_test, y_dev
def predict_and_eval(model, X_test, y_test):
    predicted = model.predict(X_test)    
    return metrics.accuracy_score(y_test, predicted)
