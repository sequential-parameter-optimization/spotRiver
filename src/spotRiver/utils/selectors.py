from river.tree.splitter import EBSTSplitter, QOSplitter
from river.linear_model import Perceptron
from river.linear_model import LinearRegression
from numpy import power


def select_splitter(i):
    if i == 0:
        return EBSTSplitter()
    else:
        return QOSplitter()


def select_leaf_prediction(i):
    if i not in range(3):
        raise ValueError('{i} wrong, use int values between 0 and 2.'.format(i=repr(i)))
    elif i == 0:
        return "mean"
    elif i == 1:
        return "adaptive"
    else:
        return "model"


def select_leaf_model(i):
    if i not in range(2):
        raise ValueError('{i} wrong, use int values between 0 and 1.'.format(i=repr(i)))
    elif i == 0:
        return LinearRegression()
    else:
        return Perceptron()


def select_max_depth(i):
    return power(10, i)
