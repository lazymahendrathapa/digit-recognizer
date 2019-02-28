import numpy as np
import model_evaluate


class LogisticRegression:

    def __init__(self, num_of_feature, lr=0.001, iteration=10000):

        self.lr = lr
        self.iteration = iteration
        self.num_of_feature = num_of_feature
        self.weights = np.random.rand(self.num_of_feature + 1)

    def loss_function(self, y_true, y_pred, eplison=0.000001):
        error = y_true * np.log(y_pred + eplison) + (1 - y_true) * np.log(1 - y_pred + eplison)

        return -1.0 * error.mean()

    def decision_boundry(self, y_pred):

        y = y_pred.copy()

        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        return y

    def sigmoid_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def calculate_gradient(self, x_true, y_true, y_pred):

        gradients = np.dot(x_true.T, y_pred - y_true)

        return gradients / len(y_true)

    def predict(self, x_pred):
        summation = x_pred.dot(self.weights)
        y_pred = self.sigmoid_activation(summation)
        return y_pred

    def model_train(self, x_train, y_train, x_val, y_val):

        # add bias
        x_train = np.insert(x_train, 0, 1, axis=1)
        x_val = np.insert(x_val, 0, 1, axis=1)

        for i in range(1, self.iteration + 1):

            y_pred = self.predict(x_train)
            train_loss = self.loss_function(y_train, y_pred)

            gradients = self.calculate_gradient(x_train, y_train, y_pred)

            self.weights -= gradients * self.lr

            if i % 100 == 0:

                y_val_pred = self.predict(x_val)
                val_loss = self.loss_function(y_val, y_val_pred)

                train_evals = model_evaluate.evaluate(y_train, self.decision_boundry(y_pred))
                val_evals = model_evaluate.evaluate(y_val, self.decision_boundry(y_val_pred))

                print("Epoch: {}".format(i))
                print("Train Loss: {}, Train Accuracy: {}".format(train_loss, train_evals['accuracy']))
                print("Val Loss: {}, Val Accuracy: {}".format(val_loss, val_evals['accuracy']))

        print('{:-^65}'.format(''))

    def model_predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        y_test = self.predict(x_test)
        return y_test
