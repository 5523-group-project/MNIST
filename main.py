import matplotlib.pyplot as plt
import numpy as np
import sklearn.base
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier





def classify(clf):
    digits = datasets.fetch_openml(name= "mnist_784")
    n_samples = len(digits.data)
    data = digits.data.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.25, shuffle=False)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def indep_classifier(clf):
    sub_classifiers = []
    for i in range(10):
        sub_classifiers.append(sklearn.base.clone(clf))

    digits = datasets.fetch_openml(name= "mnist_784")
    _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    n_samples = len(digits.data)
    data = digits.data.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.50, shuffle=False)
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
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
    multi_layer = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(), random_state=1, max_iter=10000)
    print("Lin perceptron {}".format(classify(multi_layer)))


def mlp_binary():
    clf = sklearn.neural_network.MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(500),
                                              random_state=1, max_iter=10000)
    print("Binary MLP {}".format(indep_classifier(clf)))


def mlp_sgd():
    clf = sklearn.neural_network.MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(500),learning_rate='adaptive',
                                               random_state=1, max_iter=10000)
    print("Non-binary MLP {}".format(classify(clf)))
    return clf

def mlp_lbfgs():
    clf = sklearn.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500),
                                               random_state=1, max_iter=10000)
    print("Non-binary MLP {}".format(classify(clf)))
    return clf

def mlp_adam():
    clf = sklearn.neural_network.MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(500), activation='tanh',
                                               random_state=1, max_iter=10000)
    print("Non-binary MLP {}".format(classify(clf)))
    return clf




if __name__ == "__main__":
    #perceptron_binary()
    #perceptron()
    #mlp_binary()
    mlp = mlp_sgd()
    mlp = mlp_lbfgs()
    mlp = mlp_adam()


    
