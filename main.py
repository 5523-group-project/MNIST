import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def confusion_matrix(classifier, name="Unnamed Classifier"):
    digits = datasets.load_digits()
    _, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix for {}".format(name))

    plt.show()


if __name__ == "__main__":
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
    confusion_matrix(clf, "Test")
