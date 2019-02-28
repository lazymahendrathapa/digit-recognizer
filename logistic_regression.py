import numpy as np


class LogisticRegression:

    def __init__(self, num_of_feature, lr=0.01, iteration=500):

        self.lr = lr
        self.iteration = iteration
        self.num_of_feature = num_of_feature
        self.weights = np.random.rand(self.num_of_feature + 1)

    def loss_function(self, y_true, y_pred, eplison=0.000001):
        error = y_true * np.log(y_pred + eplison) + (1 - y_true) * np.log(1 - y_pred + eplison)

        return -1.0 * error.mean()

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_gradient(self, x_true, y_true, y_pred):

        gradients = np.dot(x_true.T, y_pred - y_true)

        return gradients / len(y_true)

    def predict(self, x_pred):
        summation = x_pred.dot(self.weights)
        y_pred = self.sigmoid_activation(summation)
        return y_pred

    def model_train(self, x_true, y_true):
        # add bias
        x_true = np.insert(x_true, 0, 1, axis=1)

        for i in range(self.iteration):

            y_pred = self.predict(x_true)
            loss = self.loss_function(y_true, y_pred)
            gradients = self.calculate_gradient(x_true, y_true, y_pred)

            self.weights -= gradients * self.lr

            print(loss)

    def model_predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        y_test = self.predict(x_test)
        return y_test
