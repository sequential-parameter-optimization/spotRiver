from pandas import DataFrame
from pandas import concat
from pandas import Series
from datetime import datetime
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(diff, memory, time):
    """
    Calculate evaluation metrics for a time series forecast model.

    Args:
        diff (numpy.ndarray): Array of differences between actual values and predicted values.
        memory (float): Memory usage in megabytes.
        time (float): Computation time in seconds.

    Returns:
        dict: A dictionary of evaluation metrics.

    Evaluation metrics:
        - AIC (None): Akaike Information Criterion
        - BIC (None): Bayesian Information Criterion
        - RMSE: Root Mean Squared Error
        - MAE: Mean Absolute Error
        - AbsDiff: Absolute Difference
        - Underestimation: Total sum of positive differences (where actual value > predicted value)
        - Overestimation: Total sum of negative differences (where actual value < predicted value)
        - MaxResidual: Maximum absolute difference
        - Memory (MB): Memory usage in megabytes
        - CompTime (s): Computation time in seconds
    """
    pos_sum = np.sum(diff[diff > 0])
    neg_sum = np.sum(diff[diff < 0])
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    res_dict = {
        "AIC": None,
        "BIC": None,
        "RMSE": np.round(rmse, 2),
        "MAE": np.round(mae, 2),
        "AbsDiff": np.round(np.sum(np.abs(diff)), 2),
        "Underestimation": np.round(pos_sum, 2),
        "Overestimation": np.round(np.abs(neg_sum), 2),
        "MaxResidual": np.round(np.max(np.abs(diff)), 2),
        "Memory (MB)": np.round(memory, 4),
        "CompTime (s)": np.round(time, 4),
    }
    return res_dict


def eval_bml(train=None, test=None, horizon=None, model=None):
    df_eval = DataFrame(
        columns=[
            "RMSE",
            "MAE",
            "AbsDiff",
            "Underestimation",
            "Overestimation",
            "MaxResidual",
            "Memory (MB)",
            "CompTime (s)",
        ]
    )
    series_preds = Series([])
    series_diffs = Series([])
    start = datetime.now()
    tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    df_eval.loc[0] = Series(
        {
            "RMSE": None,
            "MAE": None,
            "AbsDiff": None,
            "Underestimation": None,
            "Overestimation": None,
            "MaxResidual": None,
            "Memory (MB)": np.round(peak / 10**6, 4),
            "CompTime (s)": np.round(time, 4),
        }
    )
    if horizon is None:
        for i in range(0, len(test)):
            series_preds, series_diffs = eval_one(
                df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
            )
    if horizon is not None:
        if len(test) % horizon == 0:
            for i in range(0, (int(len(test) / horizon) - 1)):
                series_preds, series_diffs = eval_one(
                    df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
                )
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
        if len(test) % horizon != 0:
            length = np.floor(len(test) / horizon)
            for i in range(0, (int(length))):
                series_preds, series_diffs = eval_one(
                    df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
                )
            series_preds, series_diffs = eval_one(
                df_eval, length, model, horizon, test, series_preds, series_diffs, is_last=True
            )
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs


def eval_one(df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False):
    start = datetime.now()
    tracemalloc.start()
    if is_last:
        forecast = model.predict(test.iloc[int(i * horizon) : int((i * horizon + len(test) % horizon - 1)), :-1])
    else:
        forecast = model.predict(test.iloc[i * horizon : (i + 1) * horizon, :-1])
    preds = Series(forecast)
    if is_last:
        diffs = test.iloc[int(i * horizon) : int((i * horizon + len(test) % horizon - 1)), -1].values - preds
    else:
        diffs = test.iloc[i * horizon : (i + 1) * horizon, -1].values - preds
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    if is_last:
        df_eval.loc[int(i)] = Series(evaluate_model(diffs, (peak / 10**6), time))
    else:
        df_eval.loc[i + 1] = Series(evaluate_model(diffs, (peak / 10**6), time))
    series_preds = concat([series_preds, preds])
    series_diffs = concat([series_diffs, diffs])
    return series_preds, series_diffs


def plot_results(df_eval=None, metric=False, df_true=None, real_vs_predict=False):
    if df_eval is not None and metric:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16, 5), constrained_layout=True, sharex=True)
        ax1.plot(df_eval.index, df_eval["MAE"])
        ax1.set_title("Mean Absolute Error")
        ax2.plot(df_eval.index, df_eval["Memory (MB)"])
        ax2.set_title("Memory (MB)")
        ax3.plot(df_eval.index, df_eval["CompTime (s)"])
        ax3.set_title("Computation time (s)")
    if df_true is not None and real_vs_predict:
        plt.figure(figsize=(16, 5))
        plt.plot(df_true.index, df_true["Vibration"], label="Actual")
        plt.plot(df_true.index, df_true["Prediction"], label="Prediction")
        plt.title("Actual vs Prediction")
        plt.legend()
        plt.show()
