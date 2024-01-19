import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from river import preprocessing
from river.forest import AMFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.linear_model import LogisticRegression
from math import inf
import pylab
from spotRiver.data.river_hyper_dict import RiverHyperDict
from spotRiver.utils.data_conversion import convert_to_df
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.fun.hyperriver import HyperRiver
from spotRiver.data.selector import data_selector
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
from spotPython.plot.validation import plot_roc_from_dataframes
from spotPython.plot.validation import plot_confusion_matrix
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import set_control_key_value

# from spotPython.hyperparameters.values import modify_hyper_parameter_levels
from spotPython.hyperparameters.values import get_one_core_model_from_X
from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard
from spotPython.hyperparameters.values import set_control_hyperparameter_value
from spotPython.utils.init import design_control_init, surrogate_control_init


def run_spot_river_experiment(
    MAX_TIME=1,
    INIT_SIZE=5,
    PREFIX="0000-river",
    horizon=1,
    n_total=None,
    perc_train=0.6,
    oml_grace_period=None,
    data_set="Phishing",
    target="is_phishing",
    filename="PhishingData.csv",
    directory="./userData",
    n_samples=1_250,
    n_features=9,
    converters={
        "empty_server_form_handler": float,
        "popup_window": float,
        "https": float,
        "request_from_other_domain": float,
        "anchor_from_other_domain": float,
        "is_popular": float,
        "long_url": float,
        "age_of_domain": int,
        "ip_in_url": int,
        "is_phishing": lambda x: x == "1",
    },
    parse_dates=None,
    prepmodel="StandardScaler",
    coremodel="AMFClassifier",
    log_level=50,
) -> spot.Spot:
    """Runs a spot experiment with the river package.

    Args:
        MAX_TIME (int, optional):
            Maximum time in seconds. Defaults to 1.
        INIT_SIZE (int, optional):
            Initial size of the design. Defaults to 5.
        PREFIX (str, optional):
            Prefix for the experiment name. Defaults to "0000-river".
        horizon (int, optional):
            Horizon for the evaluation. Defaults to 1.
        n_total (int, optional):
            Number of samples in the data set. Defaults to None, i.e., the
            full data set is used.
        perc_train (float, optional):
            Percentage of training samples. Defaults to 0.6.
        oml_grace_period (int, optional):
            Grace period for the online machine learning. Defaults to None.
        data_set (str, optional):
            Data set to use. Defaults to "Phishing".
        filename (str, optional):
            Name of the data file to read. Defaults to "user_data.csv".
        directory (str, optional):
            Directory where the data file is located. Defaults to "./userData".
        target (str, optional):
            Name of the target column in the data file. Defaults to "Consumption".
        n_features (int, optional):
            Number of features in the data file. Defaults to 1.
        converters (dict, optional):
            Dictionary of functions for converting data values in certain columns. Defaults to {"Consumption": float}.
        parse_dates (dict, optional):
            Dictionary of functions for parsing data values in certain columns. Defaults to {"Time": "%Y-%m-%d %H:%M:%S%z"}.
        prepmodel (str, optional):
            Name of the preprocessing model. Defaults to "StandardScaler".
        coremodel (str, optional):
            Name of the core model. Defaults to "AMFClassifier".
        log_level (int, optional):
            Log level. Defaults to 50.
    """
    fun_control = fun_control_init(
        PREFIX=PREFIX, TENSORBOARD_CLEAN=True, max_time=MAX_TIME, fun_evals=inf, tolerance_x=np.sqrt(np.spacing(1))
    )

    dataset, n_samples = data_selector(
        data_set,
        filename=filename,
        directory=directory,
        target=target,
        n_features=n_features,
        n_samples=n_samples,
        converters=converters,
        parse_dates=parse_dates,
    )
    # target_column is the name of the target column in the resulting data frame df:
    target_column = "y"
    df = convert_to_df(dataset, target_column=target_column, n_total=n_total)
    df.columns = [f"x{i}" for i in range(1, dataset.n_features + 1)] + ["y"]
    df["y"] = df["y"].astype(int)
    # update n_samples to the actual number of samples in the data set, because n_total might be smaller than n_samples which results in a smaller data set:
    n_samples = len(df)
    n_train = int(perc_train * n_samples)
    train = df[:n_train]
    print(f"train = {train.describe(include='all')}")
    test = df[n_train:]
    print(f"test = {test.describe(include='all')}")

    # if oml_grace_period is None set it to n_train:
    if oml_grace_period is None:
        oml_grace_period = n_train

    if prepmodel == "StandardScaler":
        prep_model = preprocessing.StandardScaler()
    elif prepmodel == "MinMaxScaler":
        prep_model = preprocessing.MinMaxScaler()
    else:
        prep_model = None

    weights = np.array([-1, 1 / 1000, 1 / 1000]) * 10_000.0
    weight_coeff = 1.0

    fun_control.update(
        {
            "train": train,
            "oml_grace_period": oml_grace_period,
            "test": test,
            "n_samples": n_samples,
            "target_column": target_column,
            "prep_model": prep_model,
            "horizon": horizon,
            "oml_grace_period": oml_grace_period,
            "weights": weights,
            "weight_coeff": weight_coeff,
            "metric_sklearn": accuracy_score,
            "log_level": log_level,
        }
    )

    if coremodel == "LogisticRegression":
        add_core_model_to_fun_control(
            core_model=LogisticRegression,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
        # modify_hyper_parameter_bounds(fun_control, "l2", bounds=[0.0, 0.01])
        # Note (from the River documentation):
        # For now, only one type of penalty can be used. The joint use of L1 and L2 is not explicitly supported.
        # Therefore, we set l1 bounds to 0.0:
        # modify_hyper_parameter_bounds(fun_control, "l1", bounds=[0.0, 0.0])
        # modify_hyper_parameter_levels(fun_control, "optimizer", ["SGD"])
    elif coremodel == "AMFClassifier":
        add_core_model_to_fun_control(
            core_model=AMFClassifier, fun_control=fun_control, hyper_dict=RiverHyperDict, filename=None
        )
        # modify_hyper_parameter_bounds(fun_control, "n_estimators", bounds=[2, 20])
        # modify_hyper_parameter_bounds(fun_control, "step", bounds=[0.5, 2])

        set_control_hyperparameter_value(fun_control, "n_estimators", [2, 10])
        set_control_hyperparameter_value(fun_control, "step", [0.5, 2])
    elif coremodel == "HoeffdingAdaptiveTreeClassifier":
        add_core_model_to_fun_control(
            core_model=HoeffdingAdaptiveTreeClassifier,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
    else:
        raise ValueError(
            "core_model must be 'LogisticRegression' or 'AMFClassifier' or 'HoeffdingAdaptiveTreeClassifier'"
        )

    X_start = get_default_hyperparameters_as_array(fun_control)
    fun = HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon

    design_control = design_control_init()
    set_control_key_value(control_dict=design_control, key="init_size", value=INIT_SIZE, replace=True)

    surrogate_control = surrogate_control_init(noise=True, n_theta=2)

    p_open = start_tensorboard()
    print(fun_control)

    spot_tuner = spot.Spot(
        fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control
    )
    spot_tuner.run(X_start=X_start)

    # stop_tensorboard(p_open)
    return spot_tuner, fun_control


def compare_tuned_default(spot_tuner, fun_control) -> None:
    X = spot_tuner.to_all_dim(spot_tuner.min_X.reshape(1, -1))
    print(f"X = {X}")
    model_spot = get_one_core_model_from_X(X, fun_control)
    df_eval_spot, df_true_spot = eval_oml_horizon(
        model=model_spot,
        train=fun_control["train"],
        test=fun_control["test"],
        target_column=fun_control["target_column"],
        horizon=fun_control["horizon"],
        oml_grace_period=fun_control["oml_grace_period"],
        metric=fun_control["metric_sklearn"],
    )
    X_start = get_default_hyperparameters_as_array(fun_control)
    model_default = get_one_core_model_from_X(X_start, fun_control)
    df_eval_default, df_true_default = eval_oml_horizon(
        model=model_default,
        train=fun_control["train"],
        test=fun_control["test"],
        target_column=fun_control["target_column"],
        horizon=fun_control["horizon"],
        oml_grace_period=fun_control["oml_grace_period"],
        metric=fun_control["metric_sklearn"],
    )

    df_labels = ["default", "spot"]

    # First Plot

    plot_bml_oml_horizon_metrics(
        df_eval=[df_eval_default, df_eval_spot],
        log_y=False,
        df_labels=df_labels,
        metric=fun_control["metric_sklearn"],
        filename=None,
        show=False,
    )
    plt.figure(1)

    # Second Plot
    plot_roc_from_dataframes(
        [df_true_default, df_true_spot],
        model_names=["default", "spot"],
        target_column=fun_control["target_column"],
        show=False,
    )
    plt.figure(2)
    # Third Plot

    plot_confusion_matrix(
        df=df_true_default,
        title="Default",
        y_true_name=fun_control["target_column"],
        y_pred_name="Prediction",
        show=False,
    )
    plt.figure(2)
    # Fourth Plot

    plot_confusion_matrix(
        df=df_true_spot, title="Spot", y_true_name=fun_control["target_column"], y_pred_name="Prediction", show=False
    )
    plt.figure(3)

    plt.show()  # Display all four plots simultaneously


def parallel_plot(spot_tuner):
    fig = spot_tuner.parallel_plot()
    fig.show()


def contour_plot(spot_tuner):
    spot_tuner.plot_important_hyperparameter_contour(show=False)
    pylab.show()


def importance_plot(spot_tuner):
    plt.figure()
    spot_tuner.plot_importance(show=False)
    plt.show()


def progress_plot(spot_tuner):
    spot_tuner.plot_progress(show=False)
    plt.show()
