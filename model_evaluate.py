import numpy as np


def evaluate(y_true, y_pred):

    results = {}
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    results['accuracy'] = accuracy * 100

    return results
