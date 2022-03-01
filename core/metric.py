import numpy as np


def perc(y_pred, y_true, threshold=0.2):
    """
    Calculates metric
    :param y_pred: predictions
    :param y_true: labels
    :param threshold: threshold to compare
    :return: metric value [0, 1]
    """
    residual = abs((y_true - y_pred).astype("float"))
    lin = np.where((residual <= y_true * threshold), 1, 0)

    return sum(lin)/len(y_true)