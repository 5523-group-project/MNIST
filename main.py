import matplotlib.pyplot as plt
import numpy as np
import sklearn.base
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from process import process


def classify(clf):
    digits = datasets.load_digits()
    _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    # disp.figure_.suptitle("Confusion Matrix for {}".format(name))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def indep_classifier(clf):
    sub_classifiers = []
    for i in range(10):
        sub_classifiers.append(sklearn.base.clone(clf))

    digits = datasets.load_digits()
    _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    for i in range(10):
        y_train_i = np.equal(y_train, i).astype(np.float32)
        clf: MLPClassifier = sub_classifiers[i]
        clf.fit(X_train, y_train_i)

    y_test_n = len(y_test)
    predictions_arr = np.zeros((y_test_n, 10))

    for i in range(10):
        clf = sub_classifiers[i]
        predicted_prob = clf.predict(X_test).reshape(-1)
        predictions_arr[:, i] = predicted_prob

    predictions = np.argmax(predictions_arr, axis=1)
    correct_count = np.sum(predictions == y_test)
    accuracy = correct_count / y_test_n
    return accuracy


def perceptron_binary():
    """
    Trains 10 perceptrons (for 10 possible classes) and takes the one that is most confident.
    This makes weights for each independent
    """
    single_layer = sklearn.neural_network.MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(),
                                                       random_state=1, max_iter=10000)
    print("Binary lin perceptron {}".format(indep_classifier(single_layer)))


def perceptron():
    """
    one perceptron but weights should still be fairly independent as no hidden layer
    """
    multi_layer = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(), random_state=1, max_iter=10000)
    print("Lin perceptron {}".format(classify(multi_layer)))


def mlp_binary():
    clf = sklearn.neural_network.MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100),
                                              random_state=1, max_iter=10000)
    print("Binary MLP {}".format(indep_classifier(clf)))


def mlp():
    clf = sklearn.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100),
                                               random_state=1, max_iter=10000)
    print("Non-binary MLP {}".format(classify(clf)))
    return clf


if __name__ == "__main__":
    # perceptron_binary()
    # perceptron()
    # mlp_binary()
    mlp = mlp()

    images = process("images")
    predictions = mlp.predict(images)
    print("predictions: ", predictions)
