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


def transform_power_10(i):
    return power(10, i)


def apply_selectors(d: dict):
    # Apply only if the key is present
    if "splitter" in d:
        d["splitter"] = select_splitter(d["splitter"])
    if "leaf_prediction" in d:
        d["leaf_prediction"] = select_leaf_prediction(d["leaf_prediction"])
    if "leaf_model" in d:
        d["leaf_model"] = select_leaf_model(d["leaf_model"])
    if "max_depth" in d:
        d["max_depth"] = transform_power_10(d["max_depth"])
    if "memory_estimate_period" in d:
        d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    return d
