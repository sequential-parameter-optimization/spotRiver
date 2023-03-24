from river.tree.splitter import EBSTSplitter, QOSplitter, TEBSTSplitter, GaussianSplitter, HistogramSplitter
from river.linear_model import LinearRegression, PARegressor, Perceptron
from numpy import power
from spotPython.hyperparameters.categorical import find_closest_key
from spotPython.utils.convert import class_for_name
import river.tree.splitter
import river.linear_model


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


def select_split_criterion_classifier(i):
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
        return GaussianSplitter()
    else:
        return HistogramSplitter()


def old_apply_selectors(d: dict, core_model_name: str, hyper_dict: dict):
    if core_model_name == "HoeffdingAdaptiveTreeRegressor":
        # Apply only if the key is present
        if "splitter" in d:
            d["splitter"] = select_splitter(d["splitter"])
        if "leaf_prediction" in d:
            d["leaf_prediction"] = find_closest_key(d["leaf_prediction"], hyper_dict["leaf_prediction"])
        if "leaf_model" in d:
            d["leaf_model"] = select_leaf_model(d["leaf_model"])
        if "max_depth" in d:
            d["max_depth"] = transform_power_2(d["max_depth"])
        if "memory_estimate_period" in d:
            d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    # classifier:
    if core_model_name == "HoeffdingAdaptiveTreeClassifier":
        if "split_criterion" in d:
            d["split_criterion"] = select_split_criterion_classifier(d["split_criterion"])
        if "leaf_prediction" in d:
            d["leaf_prediction"] = select_leaf_prediction_classifier(d["leaf_prediction"])
        if "splitter" in d:
            d["splitter"] = select_splitter_classifier(d["splitter"])
        if "max_depth" in d:
            d["max_depth"] = transform_power_2(d["max_depth"])
        if "memory_estimate_period" in d:
            d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    if core_model_name == "HoeffdingTreeRegressor":
        if "leaf_prediction" in d:
            d["leaf_prediction"] = find_closest_key(d["leaf_prediction"], hyper_dict["leaf_prediction"])
        if "splitter" in d:
            splitter_class = class_for_name(
                "river.tree.splitter", find_closest_key(d["splitter"], hyper_dict["splitter"])
            )
            d["splitter"] = splitter_class()
        if "leaf_model" in d:
            leaf_model_class = class_for_name(
                "river.linear_model", find_closest_key(d["leaf_model"], hyper_dict["leaf_model"])
            )
            d["leaf_model"] = leaf_model_class()
        if "max_depth" in d:
            d["max_depth"] = transform_power_2(d["max_depth"])
        if "memory_estimate_period" in d:
            d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    return d


def apply_selectors(d: dict, core_model_name: str, hyper_dict: dict):
    if core_model_name == "HoeffdingAdaptiveTreeRegressor":
        # Apply only if the key is present
        if "splitter" in d:
            d["splitter"] = select_splitter(d["splitter"])
        if "leaf_prediction" in d:
            d["leaf_prediction"] = find_closest_key(d["leaf_prediction"], hyper_dict["leaf_prediction"])
        if "leaf_model" in d:
            d["leaf_model"] = select_leaf_model(d["leaf_model"])
        if "max_depth" in d:
            d["max_depth"] = transform_power_2(d["max_depth"])
        if "memory_estimate_period" in d:
            d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    # classifier:
    if core_model_name == "HoeffdingAdaptiveTreeClassifier":
        if "split_criterion" in d:
            d["split_criterion"] = select_split_criterion_classifier(d["split_criterion"])
        if "leaf_prediction" in d:
            d["leaf_prediction"] = select_leaf_prediction_classifier(d["leaf_prediction"])
        if "splitter" in d:
            d["splitter"] = select_splitter_classifier(d["splitter"])
        if "max_depth" in d:
            d["max_depth"] = transform_power_2(d["max_depth"])
        if "memory_estimate_period" in d:
            d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    if core_model_name == "HoeffdingTreeRegressor":
        if "leaf_prediction" in d:
            d["leaf_prediction"] = find_closest_key(d["leaf_prediction"], hyper_dict["leaf_prediction"])
        if "splitter" in d:
            splitter_class = class_for_name(
                "river.tree.splitter", find_closest_key(d["splitter"], hyper_dict["splitter"])
            )
            d["splitter"] = splitter_class()
        if "leaf_model" in d:
            leaf_model_class = class_for_name(
                "river.linear_model", find_closest_key(d["leaf_model"], hyper_dict["leaf_model"])
            )
            d["leaf_model"] = leaf_model_class()
        if "max_depth" in d:
            d["max_depth"] = transform_power_2(d["max_depth"])
        if "memory_estimate_period" in d:
            d["memory_estimate_period"] = transform_power_10(d["memory_estimate_period"])
    return d
