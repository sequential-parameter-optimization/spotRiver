from river.tree.splitter import EBSTSplitter, QOSplitter, TEBSTSplitter
from river.linear_model import LinearRegression, PARegressor, Perceptron
from numpy import power


def select_splitter(i):
    if i not in range(3):
        raise ValueError("{i} wrong splitter, use int values between 0 and 2.".format(i=repr(i)))
    if i == 0:
        return EBSTSplitter()
    elif i == 1:
        return TEBSTSplitter()
    else:
        return QOSplitter()


def select_leaf_prediction(i):
    if i not in range(3):
        raise ValueError("{i} wrong leaf_prediction, use int values between 0 and 2.".format(i=repr(i)))
    elif i == 0:
        return "mean"
    elif i == 1:
        return "adaptive"
    else:
        return "model"


def select_leaf_model(i):
    if i not in range(3):
        raise ValueError("{i} wrong leaf_model, use int values between 0 and 2.".format(i=repr(i)))
    elif i == 0:
        return LinearRegression()
    elif i == 1:
        return PARegressor()
    else:
        return Perceptron()


def select_max_depth(i):
    return power(10, i)
