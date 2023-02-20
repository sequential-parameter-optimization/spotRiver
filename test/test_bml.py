import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from datetime import datetime
from numpy.testing import assert_array_equal, assert_almost_equal
from pandas.testing import assert_frame_equal
from unittest.mock import MagicMock
from spotRiver.evaluation.eval_bml import eval_bml, evaluate_model


def test_eval_bml():
    # Create some test data
    data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    df = pd.DataFrame(data)
    horizon = 3
    model = MagicMock()
    model.predict.return_value = np.array([1, 2, 3])

    # Call the eval_bml function
    df_eval, df_true, series_preds, series_diffs = eval_bml(df, horizon, model)

    # Check that the evaluation metrics are correct
    expected_eval_metrics = {
        "RMSE": 0.0,
        "MAE": 0.0,
        "AbsDiff": 0.0,
        "Underestimation": 0.0,
        "Overestimation": 0.0,
        "MaxResidual": 0.0,
        "Memory (MB)": df_eval.loc[0]["Memory (MB)"],
        "CompTime (s)": df_eval.loc[0]["CompTime (s)"]
    }
    assert_series_equal(df_eval.loc[1], pd.Series(expected_eval_metrics))

    # Check that the predicted values are correct
    expected_preds = pd.Series([1, 2, 3], index=[3, 4, 5])
    assert_series_equal(series_preds, expected_preds)

    # Check that the difference between the true and predicted values is correct
    expected_diffs = pd.Series([0, 0, 0], index=[3, 4, 5])
    assert_series_equal(series_diffs, expected_diffs)

    # Check that the true dataframe is correct
    expected_data = {
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y_pred': [np.nan, np.nan, np.nan, 1, 2, 3, 1, 2, 3, 1],
        'diff': [np.nan, np.nan, np.nan, 0, 0, 0, 6, 6, 6, -1]
    }
    expected_df_true = pd.DataFrame(expected_data)
    assert_frame_equal(df_true, expected_df_true)
from spotRiver import data
from spotPython.utils.features import get_hour_distances, get_month_distances, get_ordinal_date, get_weekday_distances


def test_features():
    """
    Test features
    """
    dataset = data.AirlinePassengers()

    for x, _ in dataset:
        assert(get_ordinal_date(x)["ordinal_date"] ==  711493)
        assert(get_hour_distances(x)["0"] == 1.0)
        assert(get_weekday_distances(x)["Saturday"] == 1.0)
        assert(get_month_distances(x)["January"] == 1.0)
        break