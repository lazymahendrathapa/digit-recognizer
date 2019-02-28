import data_loader
import logistic_regression
from collections import OrderedDict
import numpy as np
import pandas as pd


def run_multi_class_logistic_regression():
    classifiers = OrderedDict()
    # model training
    for label in range(0, 10):
        print("For Label: ", label)
        x_train, y_train, x_val, y_val = data_loader.load_train_dataset(label)
        num_of_feature = x_train.shape[1]
        classifier = logistic_regression.LogisticRegression(num_of_feature)
        classifier.model_train(x_train, y_train, x_val, y_val)
        classifiers[label] = classifier

    # model prediction
    results = []
    x_test = data_loader.load_test_dataset()
    for label, classifier in classifiers.items():
        y_test = classifier.model_predict(x_test)
        results.append(y_test)

    results = np.array(results)

    predictions = np.argmax(results, axis=0)
    df = pd.DataFrame(predictions)
    df.index += 1
    df.index.name = 'ImageId'
    df.columns = ['Label']
    df.to_csv('out/logistic_prediction.csv')


if __name__ == '__main__':
    run_multi_class_logistic_regression()
