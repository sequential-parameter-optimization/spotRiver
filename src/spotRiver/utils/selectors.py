from river.tree.splitter import EBSTSplitter, QOSplitter, TEBSTSplitter, GaussianSpitter, HistogramSplitter
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


def transform_power_2(i):
    return power(2, i)

def split_criterion_classifierl(i):
    if i not in range(3):
        raise ValueError("{i} wrong split_criterion_classifier, use int values between 0 and 2.".format(i=repr(i)))
    elif i == 0:
        return "gini"
    elif i == 1:
        return "info_gain"
    else:
        return "hellinger"

def select_leaf_prediction_classifier(i):
    if i not in range(3):
        raise ValueError("{i} wrong leaf_prediction, use int values between 0 and 2.".format(i=repr(i)))
    elif i == 0:
        return "mc"
    elif i == 1:
        return "nb"
    else:
        return "nba"

def select_splitter_classifier(i):
    if i not in range(2):
        raise ValueError("{i} wrong splitter, use int values between 0 and 2.".format(i=repr(i)))
    if i == 0:
        return GaussianSpitter()
    else:
        return HistogramSplitter()


def apply_selectors(d: dict):
    # Apply only if the key is present
    if "splitter" in d:
        d["splitter"] = select_splitter(d["splitter"])
    if "leaf_prediction" in d:
        d["leaf_prediction"] = select_leaf_prediction(d["leaf_prediction"])
    if "leaf_model" in d:
        d["leaf_model"] = select_leaf_model(d["leaf_model"])
    if "max_depth" in d:
        d["max_depth"] = transform_power_2(d["max_depth"])
    if "memory_estimate_period" in d:
        d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    # classifier:
    if "split_criterion_classifier" in d:
        d["split_criterion_classifier"] = select_split_criterion_classifier(d["split_criterion_classifier"])
    if "leaf_prediction_classifier" in d:
        d["leaf_prediction_classifier"] = select_leaf_prediction_classifier(d["leaf_prediction_classifier"])
    if "splitter_classifier" in d:
        d["splitter_classifier"] = select_splitter_classifier(d["splitter_classifier"])
    return d
