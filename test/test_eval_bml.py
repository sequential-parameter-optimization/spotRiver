import pandas as pd
import numpy as np
from river import linear_model
from river import preprocessing
from sklearn.metrics import mean_absolute_error
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

