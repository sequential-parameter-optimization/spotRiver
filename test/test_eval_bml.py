import pandas as pd
import numpy as np
from river import linear_model
from river import preprocessing
from sklearn.metrics import mean_absolute_error
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from sklearn.metrics import accuracy_score
from river import preprocessing
from river.forest import AMFClassifier
from river.datasets import Bananas
from spotRiver.data.river_hyper_dict import RiverHyperDict
from spotRiver.utils.data_conversion import convert_to_df
from spotPython.hyperparameters.values import add_core_model_to_fun_control
from spotPython.utils.init import fun_control_init
from spotPython.hyperparameters.values import modify_hyper_parameter_bounds
from spotPython.hyperparameters.values import get_one_core_model_from_X
from spotPython.hyperparameters.values import get_default_hyperparameters_as_array
from spotRiver.evaluation.eval_bml import eval_oml_horizon


def test_eval_oml_horizon():
    # create a sample model
    model = (
        preprocessing.StandardScaler() |
        linear_model.LinearRegression(intercept_lr=.5)
    )

    # create a sample train DataFrame
    train = pd.DataFrame({"x": np.arange(1, 11), "y": np.arange(2, 22, 2)})

    # create a sample test DataFrame
    test = pd.DataFrame({"x": np.arange(11, 111), "y": np.arange(22, 222, 2)})

    # set the target column
    target_column = "y"

    # set the horizon
    horizon = 5

    # set the metric
    metric = mean_absolute_error

    # evaluate the model
    res, preds = eval_oml_horizon(
        model = model,
        train = train,
        test = test,
        target_column = target_column,
        horizon = horizon,
        include_remainder = True,
        metric = metric,
        oml_grace_period = horizon,
    )

    # result should have one value for the initial model and one value for each horizon
    assert res.shape[0] == 1 + test.shape[0] // horizon
    # predictions  should be based on the test set only
    assert preds.shape == (test.shape[0], 3)

def test_eval_oml_horizon_with_default():
    PREFIX = "0000"
    fun_control = fun_control_init(
        PREFIX=PREFIX,
        TENSORBOARD_CLEAN=True)
    #fun_control
    horizon = 30
    n_samples = 5300
    n_train = 5000
    oml_grace_period = n_train
    dataset = Bananas(
    )
    target_column = "y"
    weights = np.array([- 1, 1/1000, 1/1000])*10_000.0
    df = convert_to_df(dataset, target_column=target_column, n_total=n_samples)
    df.columns = [f"x{i}" for i in range(1, dataset.n_features+1)] + ["y"]
    df["y"] = df["y"].astype(int)
    prep_model = preprocessing.StandardScaler()
    fun_control.update({"train":  df[:n_train],
                        "oml_grace_period": oml_grace_period,
                        "test":  df[n_train:],
                        "n_samples": n_samples,
                        "target_column": target_column,
                        "prep_model": prep_model,
                        "horizon": horizon,
                        "oml_grace_period": oml_grace_period,
                        "weights": weights,
                        "metric_sklearn": accuracy_score
                        })
    add_core_model_to_fun_control(core_model=AMFClassifier,
                                fun_control=fun_control,
                                hyper_dict=RiverHyperDict,
                                filename=None)
    modify_hyper_parameter_bounds(fun_control, "n_estimators", bounds=[2,20])
    modify_hyper_parameter_bounds(fun_control, "step", bounds=[0.5,2])


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
    assert df_eval_default.shape == ((n_samples - n_train) // horizon + 1,3)