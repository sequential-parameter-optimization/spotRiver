import numpy as np
import matplotlib.pyplot as plt
from math import inf
import pylab
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.fun.hyperriver import HyperRiver
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
from spotPython.plot.validation import plot_roc_from_dataframes
from spotPython.plot.validation import plot_confusion_matrix
from spotPython.utils.init import fun_control_init

# from spotPython.hyperparameters.values import modify_hyper_parameter_levels
from spotPython.spot import spot
from spotPython.utils.tensorboard import start_tensorboard
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

    X_start = get_default_hyperparameters_as_array(fun_control)
    fun = HyperRiver(log_level=fun_control["log_level"]).fun_oml_horizon

    surrogate_control = surrogate_control_init(noise=True, n_theta=2)

    p_open = start_tensorboard()
    print(fun_control)

    spot_tuner = spot.Spot(
        fun=fun, fun_control=fun_control, design_control=design_control, surrogate_control=surrogate_control
    )
    spot_tuner.run(X_start=X_start)

    # stop_tensorboard(p_open)
    return spot_tuner, fun_control


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
