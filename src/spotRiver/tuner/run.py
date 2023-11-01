import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from river import preprocessing
from river.forest import AMFClassifier
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.datasets import Bananas, CreditCard, Phishing
from math import inf
import pandas as pd
import spotRiver
from spotRiver.data.river_hyper_dict import RiverHyperDict
from spotRiver.utils.data_conversion import convert_to_df
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.fun.hyperriver import HyperRiver
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
from spotPython.plot.validation import plot_roc_from_dataframes
from spotPython.plot.validation import plot_confusion_matrix
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.utils.init import fun_control_init
from spotPython.utils.file import get_spot_tensorboard_path
from spotPython.utils.file import get_experiment_name
from spotPython.hyperparameters.values import modify_hyper_parameter_bounds
from spotPython.hyperparameters.values import get_one_core_model_from_X
from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
from spotPython.spot import spot
from spotPython.hyperparameters.values import get_var_type, get_var_name, get_bound_values
from spotPython.utils.eda import gen_design_table
from spotPython.utils.tensorboard import start_tensorboard, stop_tensorboard


def run_spot_river_experiment(
    MAX_TIME=1,
    INIT_SIZE=5,
    PREFIX="0000-river",
    horizon=1,
    n_samples=None,
    n_train=None,
    oml_grace_period=None,
    data_set="Phishing",
    prepmodel="StandardScaler",
    coremodel="AMFClassifier",
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
        n_samples (int, optional):
            Number of samples in the data set. Defaults to None, i.e., the
            full data set is used.
        n_train (int, optional):
            Number of training samples. Defaults to None.
        oml_grace_period (int, optional):
            Grace period for the online machine learning. Defaults to None.
        data_set (str, optional):
            Data set to use. Defaults to "Phishing".
    """
    experiment_name = get_experiment_name(prefix=PREFIX)
    fun_control = fun_control_init(
        spot_tensorboard_path=get_spot_tensorboard_path(experiment_name), TENSORBOARD_CLEAN=True
    )

    if data_set == "Bananas":
        dataset = Bananas()
    elif data_set == "CreditCard":
        dataset = CreditCard()
    elif data_set == "Phishing":
        dataset = Phishing()
        horizon = 1
        n_samples = 1250
        n_train = 100
        oml_grace_period = 100
    else:
        raise ValueError("data_set must be 'Bananas' or 'ConceptDriftStream'")
    target_column = "y"
    weights = np.array([-1, 1 / 1000, 1 / 1000]) * 10_000.0
    weight_coeff = 1.0
    df = convert_to_df(dataset, target_column=target_column, n_total=n_samples)
    df.columns = [f"x{i}" for i in range(1, dataset.n_features + 1)] + ["y"]
    df["y"] = df["y"].astype(int)

    if prepmodel == "StandardScaler":
        prep_model = preprocessing.StandardScaler()
    elif prepmodel == "MinMaxScaler":
        prep_model = preprocessing.MinMaxScaler()
    else:
        prep_model = None

    fun_control.update(
        {
            "train": df[:n_train],
            "oml_grace_period": oml_grace_period,
            "test": df[n_train:],
            "n_samples": n_samples,
            "target_column": target_column,
            "prep_model": prep_model,
            "horizon": horizon,
            "oml_grace_period": oml_grace_period,
            "weights": weights,
            "weight_coeff": weight_coeff,
            "metric_sklearn": accuracy_score,
        }
    )

    if coremodel == "AMFClassifier":
        add_core_model_to_fun_control(
            core_model=AMFClassifier, fun_control=fun_control, hyper_dict=RiverHyperDict, filename=None
        )
        modify_hyper_parameter_bounds(fun_control, "n_estimators", bounds=[2, 20])
        modify_hyper_parameter_bounds(fun_control, "step", bounds=[0.5, 2])
    elif coremodel == "HoeffdingAdaptiveTreeClassifier":
        add_core_model_to_fun_control(
            core_model=HoeffdingAdaptiveTreeClassifier,
            fun_control=fun_control,
            hyper_dict=RiverHyperDict,
            filename=None,
        )
    else:
        raise ValueError("core_model must be 'AMFClassifier' or 'HoeffdingAdaptiveTreeClassifier'")

    X_start = get_default_hyperparameters_as_array(fun_control)
    fun = HyperRiver(log_level=50).fun_oml_horizon
    var_type = get_var_type(fun_control)
    var_name = get_var_name(fun_control)
    lower = get_bound_values(fun_control, "lower")
    upper = get_bound_values(fun_control, "upper")

    p_open = start_tensorboard()
    spot_tuner = spot.Spot(
        fun=fun,
        lower=lower,
        upper=upper,
        fun_evals=inf,
        max_time=MAX_TIME,
        tolerance_x=np.sqrt(np.spacing(1)),
        var_type=var_type,
        var_name=var_name,
        show_progress=True,
        fun_control=fun_control,
        design_control={"init_size": INIT_SIZE},
        surrogate_control={
            "noise": False,
            "cod_type": "norm",
            "min_theta": -4,
            "max_theta": 3,
            "n_theta": len(var_name),
            "model_fun_evals": 10_000,
        },
        log_level=50,
    )
    spot_tuner.run(X_start=X_start)
    stop_tensorboard(p_open)
    # return spot_tuner and fun_control for further analysis
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
