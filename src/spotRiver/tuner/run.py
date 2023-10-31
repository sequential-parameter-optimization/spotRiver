import numpy as np
from sklearn.metrics import accuracy_score
from river import preprocessing
from river.forest import AMFClassifier
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
from spotRiver.evaluation.eval_bml import plot_bml_oml_horizon_metrics
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
    prep_model = preprocessing.StandardScaler()
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
    add_core_model_to_fun_control(
        core_model=AMFClassifier, fun_control=fun_control, hyper_dict=RiverHyperDict, filename=None
    )
    modify_hyper_parameter_bounds(fun_control, "n_estimators", bounds=[2, 20])
    modify_hyper_parameter_bounds(fun_control, "step", bounds=[0.5, 2])

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
    return spot_tuner


if __name__ == "__main__":
    tuner = run_spot_river_experiment()
