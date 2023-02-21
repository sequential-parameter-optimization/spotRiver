from pandas import DataFrame
from pandas import concat
from pandas import Series
from datetime import datetime
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import sklearn
from river import stream as river_stream


def evaluate_model(diff: np.ndarray, memory: float, time: float) -> dict:
    """
    Calculate evaluation metrics for a time series forecast model.

    Args:
        diff: Array of differences between actual values and predicted values.
            It should be a numpy array of shape (n,).
        memory: Memory usage in megabytes.
            It should be a float.
        time: Computation time in seconds.
            It should be a float.

    Returns:
        A dictionary of evaluation metrics.

    Evaluation metrics:
        - AIC: Akaike Information Criterion (None).
        - BIC: Bayesian Information Criterion (None).
        - RMSE: Root Mean Squared Error.
        - MAE: Mean Absolute Error.
        - AbsDiff: Absolute Difference.
        - Underestimation: Total sum of positive differences (where actual value > predicted value).
        - Overestimation: Total sum of negative differences (where actual value < predicted value).
        - MaxResidual: Maximum absolute difference.
        - Memory (MB): Memory usage in megabytes.
        - CompTime (s): Computation time in seconds.
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


def eval_bml(
    test: Optional[pd.DataFrame] = None,
    train: Optional[pd.DataFrame] = None,
    horizon: Optional[int] = None,
    model: Optional[object] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Evaluate the performance of a model for a given test set using a specified horizon.

    Args:
        train (pandas.DataFrame): Train data to evaluate the model.
        test (pandas.DataFrame): Test data to evaluate the model.
        horizon (int): Forecast horizon, i.e. the number of time steps to predict ahead.
        model: Trained machine learning model to be evaluated.

    Returns:
        A tuple of the following:
        - df_eval (pandas.DataFrame): Dataframe containing evaluation metrics for the model.
        - df_true (pandas.DataFrame): Dataframe with the original test data, the model predictions,
        and their difference.
        - series_preds (pandas.Series): Series with the model predictions.
        - series_diffs (pandas.Series): Series with the differences between the test data and the
        model predictions.

    The eval_bml() function evaluates the performance of a model by computing several error metrics
    on a given test set.
    The function returns a DataFrame with the error metrics, the predicted values, the differences
    between the predicted and true values, and the memory usage and computation time of the function.

    The function starts by creating an empty DataFrame df_eval with columns for the RMSE, MAE,
    Absolute Difference, Underestimation, Overestimation, MaxResidual, Memory (MB), and CompTime (s).
    The function also creates empty Series objects for series_preds and series_diffs.
    Next, the function starts the timer and memory profiler to measure the performance of the function.
    The function creates an initial row in df_eval with memory and time information.
    If horizon is not provided, the function loops over each row in the test set and calls the eval_one()
    function to compute error metrics for each row.
    If horizon is provided, the function splits the test set into multiple subsets of length horizon.
    If the length of the test set is divisible by horizon, the function loops over each subset and calls
    the eval_one() function to compute error metrics for each subset. If the length of the test set is not
    divisible by horizon, the function calls eval_one() for all complete subsets and then calls it one more
    time for the last incomplete subset.
    Finally, the function creates a DataFrame df_true by concatenating the test DataFrame with the predicted
    and difference Series objects. The function returns df_eval, df_true, series_preds, and series_diffs.

    """
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
    model.fit(train.iloc[:, :-1], train.iloc[:, -1])
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
            series_preds, series_diffs, df_eval = eval_one(
                df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
            )
    if horizon is not None:
        if len(test) % horizon == 0:
            for i in range(0, (int(len(test) / horizon) - 1)):
                series_preds, series_diffs, df_eval = eval_one(
                    df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
                )
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
        if len(test) % horizon != 0:
            length = np.floor(len(test) / horizon)
            for i in range(0, (int(length))):
                series_preds, series_diffs, df_eval = eval_one(
                    df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
                )
            series_preds, series_diffs, df_eval = eval_one(
                df_eval, length, model, horizon, test, series_preds, series_diffs, is_last=True
            )
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs


def eval_one(
    df_eval: pd.DataFrame,
    i: int,
    model: sklearn.base.BaseEstimator,
    horizon: int,
    test: pd.DataFrame,
    series_preds: pd.Series,
    series_diffs: pd.Series,
    is_last: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Evaluate the performance of a time series model for a given time period.

    Args:
    - df_eval (pandas.DataFrame): DataFrame to store the evaluation metrics.
    - i (int): Index of the current time period being evaluated.
    - model (sklearn.base.BaseEstimator): A time series model that has been fit to the training data.
    - horizon (int): The forecasting horizon, i.e. the number of periods to forecast.
    - test (pandas.DataFrame): The test data containing the time series to be forecast.
    - series_preds (pandas.Series): A Series containing the model's forecasts for the time series up to
    the current time period.
    - series_diffs (pandas.Series): A Series containing the differences between the actual values
    and the model's forecasts for the time series up to the current time period.
    - is_last (bool, optional): A boolean indicating whether the current time period is the last period
    to be evaluated. Defaults to False.

    Returns:
    - Tuple[pandas.Series, pandas.Series, pandas.DataFrame]: A tuple containing the following:
        - series_preds (pandas.Series): A Series containing the model's forecasts for the time series up to and
        including the current time period.
        - series_diffs (pandas.Series): A Series containing the differences between the actual values and the
        model's forecasts for the time series up to and including the current time period.
        - df_eval (pandas.DataFrame): The DataFrame storing the evaluation metrics.
    """

    """
    Evaluate the performance of a time series model for a given time period.

    Args:
    - df_eval (pandas.DataFrame): DataFrame to store the evaluation metrics.
    - i (int): Index of the current time period being evaluated.
    - model (sklearn.base.BaseEstimator): A time series model that has been fit to the training data.
    - horizon (int): The forecasting horizon, i.e. the number of periods to forecast.
    - test (pandas.DataFrame): The test data containing the time series to be forecast.
    - series_preds (pandas.Series): A Series containing the model's forecasts for the time series up to
    the current time period.
    - series_diffs (pandas.Series): A Series containing the differences between the actual values
    and the model's forecasts for the time series up to the current time period.
    - is_last (bool, optional): A boolean indicating whether the current time period is the last period
    to be evaluated. Defaults to False.

    Returns:
    - series_preds (pandas.Series): A Series containing the model's forecasts for the time series up to and
    including the current time period.
    - series_diffs (pandas.Series): A Series containing the differences between the actual values and the
    model's forecasts for the time series up to and including the current time period.
    - df_eval
    """
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
    return series_preds, series_diffs, df_eval


def eval_bml_landmark(
    train: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
    horizon: Optional[int] = None,
    model: Optional[object] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Evaluates the model on a set of landmarks, and returns the evaluation results
    dataframe, true values dataframe, predicted values series, and difference series.

    Parameters:
        - train (pd.DataFrame): Dataframe containing the training data.
        - test (pd.DataFrame): Dataframe containing the test data.
        - horizon (int): The number of time steps to forecast for each landmark.
        - model (object): An object that implements the model
    """
    df_eval = pd.DataFrame(
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
    series_preds = pd.Series([])
    series_diffs = pd.Series([])
    start = datetime.now()
    tracemalloc.start()
    model.fit(train.iloc[:, :-1], train.iloc[:, -1])
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    df_eval.loc[0] = pd.Series(
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
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict(np.array(test.iloc[i, :-1]).reshape(1, -1))
            preds = pd.Series(forecast, index=[i])
            diffs = test.iloc[i, -1] - preds
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
            series_preds = pd.concat([series_preds, preds])
            series_diffs = pd.concat([series_diffs, diffs])
    if horizon is not None:
        if len(test) % horizon == 0:
            landmark_data = train.copy()
            for i in range(0, (int(len(test) / horizon) - 1)):
                series_preds, series_diffs, df_eval, landmark_data = eval_one_landmark_or_window(
                    landmark_data, df_eval, i, model, horizon, test, series_preds, series_diffs
                )
        if len(test) % horizon != 0:
            landmark_data = train.copy()
            length = np.floor(len(test) / horizon)
            for i in range(0, (int(length))):
                series_preds, series_diffs, df_eval, landmark_data = eval_one_landmark_or_window(
                    landmark_data, df_eval, i, model, horizon, test, series_preds, series_diffs
                )
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict(
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), :-1]
            )
            preds = pd.Series(forecast)
            diffs = (
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), -1].values - preds
            )
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[int(length)] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
            series_preds = pd.concat([series_preds, preds])
            series_diffs = pd.concat([series_diffs, diffs])
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs


def eval_one_landmark_or_window(
    data: pd.DataFrame,
    df_eval: pd.DataFrame,
    i: int,
    model: object,
    horizon: int,
    test: pd.DataFrame,
    series_preds: pd.Series,
    series_diffs: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a machine learning model's performance on a single landmark, or subset of data, by training on that landmark
    and making predictions on a horizon of data following that landmark.

    Args:
    - data: A pandas DataFrame containing the landmark or window data.
    - df_eval: A pandas DataFrame to store the evaluation metrics for the model.
    - i: An integer representing the index of the landmark.
    - model: A machine learning model that has a `predict` and `fit` method.
    - horizon: An integer representing the number of time steps to forecast.
    - test: A pandas DataFrame containing the test data.
    - series_preds: A pandas Series to store the model's predictions.
    - series_diffs: A pandas Series to store the differences between the model's predictions and the actual values.

    Returns:
    - series_preds: A pandas Series containing the model's predictions.
    - series_diffs: A pandas Series containing the differences between the model's predictions and the actual values.
    - df_eval: A pandas DataFrame containing the evaluation metrics for the model.
    - data: A pandas DataFrame containing the concatenated landmark or window data and horizon of data used to train the model.
    """
    # Record start time and memory usage
    start = datetime.now()
    tracemalloc.start()

    # Make predictions on the horizon of data following the landmark
    forecast = model.predict(test.iloc[i * horizon : (i + 1) * horizon, :-1])
    preds = pd.Series(forecast)
    diffs = test.iloc[i * horizon : (i + 1) * horizon, -1].values - preds

    # Concatenate the landmark data and the current horizon of data and use it to train the model
    data = pd.concat([data, test.iloc[i * horizon : (i + 1) * horizon, :]])
    model.fit(data.iloc[:, :-1], data.iloc[:, -1])

    # Record end time and memory usage, and compute the time taken to evaluate the model
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()

    # Compute evaluation metrics and store results in the df_eval DataFrame
    df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))

    # Concatenate predicted values and differences with the existing series_preds and series_diffs Series
    series_preds = pd.concat([series_preds, preds])
    series_diffs = pd.concat([series_diffs, diffs])

    # Return the updated Series and DataFrame objects
    return series_preds, series_diffs, df_eval, data


def eval_bml_window(
    train: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
    horizon: Optional[int] = None,
    model: Optional[object] = None,
) -> tuple:
    """
    Evaluates a machine learning model's predictions on a time series dataset using a fixed window or a
    sliding window approach. Computes various performance metrics and returns them in a DataFrame.

    Args:
        train (pd.DataFrame): The training set.
        test (pd.DataFrame): The test set.
        horizon (int): If not None, the sliding window approach is used with the specified window size.
        model (object): A fitted machine learning model.

    Returns:
        tuple: A tuple containing the evaluation DataFrame, the true values DataFrame, a series of predicted values,
               and a series of difference values.
    """
    df_eval = pd.DataFrame(
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
    series_preds = pd.Series([])
    series_diffs = pd.Series([])
    start = datetime.now()
    tracemalloc.start()
    model.fit(train.iloc[:, :-1], train.iloc[:, -1])
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    df_eval.loc[0] = pd.Series(
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
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict(np.array(test.iloc[i, :-1]).reshape(1, -1))
            preds = pd.Series(forecast, index=[i])
            diffs = test.iloc[i, -1] - preds
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
            series_preds = pd.concat([series_preds, preds])
            series_diffs = pd.concat([series_diffs, diffs])
    if horizon is not None:
        if len(test) % horizon == 0:
            shifting_window_data = train.copy()
            for i in range(0, (int(len(test) / horizon) - 1)):
                series_preds, series_diffs, df_eval = eval_one_landmark_or_window(
                    df_eval, i, model, horizon, test, series_preds, series_diffs
                )
        if len(test) % horizon != 0:
            shifting_window_data = train.copy()
            length = np.floor(len(test) / horizon)
            for i in range(0, (int(length))):
                series_preds, series_diffs, df_eval, shifting_window_data = eval_one_landmark_or_window(
                    shifting_window_data, df_eval, i, model, horizon, test, series_preds, series_diffs
                )
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict(
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), :-1]
            )
            preds = pd.Series(forecast)
            diffs = (
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), -1].values - preds
            )
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[int(length)] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
            series_preds = pd.concat([series_preds, preds])
            series_diffs = pd.concat([series_diffs, diffs])
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs


def plot_bml_results(
    df_eval: pd.DataFrame = None, metric: bool = False, df_true: pd.DataFrame = None, real_vs_predict: bool = False
) -> None:
    """
    This function creates plots to visualize evaluation results or actual vs predicted values.
    The function does not return anything, it only generates and shows the plots.
    Args:
        df_eval (pd.DataFrame, optional): Dataframe containing evaluation results. Defaults to None.
        metric (bool, optional): If True, plot evaluation metrics. Defaults to False.
        df_true (pd.DataFrame, optional): Dataframe containing actual and predicted values. Defaults to None.
        real_vs_predict (bool, optional): If True, plot actual vs predicted values. Defaults to False.

    Returns:
        None

    The plot_results function takes in four arguments: df_eval, metric, df_true, and real_vs_predict.
    df_eval and df_true are both optional pandas DataFrames.
    If df_eval is not None and metric is True, the function plots the Mean Absolute Error, Memory usage,
    and Computation time on a 3 subplot figure.
    If df_true is not None and real_vs_predict is True, the function plots the actual and predicted values on a separate figure.

    """
    if df_eval is not None and metric:
        # create a 3 subplot figure to plot evaluation metrics
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16, 5), constrained_layout=True, sharex=True)

        # plot MAE
        ax1.plot(df_eval.index, df_eval["MAE"])
        ax1.set_title("Mean Absolute Error")

        # plot memory usage
        ax2.plot(df_eval.index, df_eval["Memory (MB)"])
        ax2.set_title("Memory (MB)")

        # plot computation time
        ax3.plot(df_eval.index, df_eval["CompTime (s)"])
        ax3.set_title("Computation time (s)")

    if df_true is not None and real_vs_predict:
        # plot actual vs predicted values
        plt.figure(figsize=(16, 5))
        plt.plot(df_true.index, df_true["Vibration"], label="Actual")
        plt.plot(df_true.index, df_true["Prediction"], label="Prediction")
        plt.title("Actual vs Prediction")
        plt.legend()
        plt.show()


def eval_oml_horizon(train=None, test=None, horizon=None, model=None):
    df_eval = pd.DataFrame(
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
    series_preds = pd.Series([])
    series_diffs = pd.Series([])
    start = datetime.now()
    tracemalloc.start()
    for xi, yi in river_stream.iter_pandas(train.iloc[:, :-1], train.iloc[:, -1]):
        model = model.learn_one(xi, yi)
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    df_eval.loc[0] = pd.Series(
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
        # i = 0
        for i, (xi, yi) in enumerate(river_stream.iter_pandas(test.iloc[:, :-1], test.iloc[:, -1])):
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict_one(xi)
            preds = pd.Series(forecast, index=[i])
            diffs = test.iloc[i, -1] - preds
            # i += 1
            model = model.learn_one(xi, yi)
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[i] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
            series_preds = pd.concat([series_preds, preds])
            series_diffs = pd.concat([series_diffs, diffs])
    if horizon is not None:
        if len(test) % horizon == 0:
            for i in range(0, (int(len(test) / horizon) - 1)):
                series_preds, series_diffs, df_eval = eval_one_oml_horizon(
                    df_eval, i, model, horizon, test, series_preds, series_diffs
                )
        if len(test) % horizon != 0:
            length = np.floor(len(test) / horizon)
            for i in range(0, (int(length))):
                series_preds, series_diffs, df_eval = eval_one_oml_horizon(
                    df_eval, i, model, horizon, test, series_preds, series_diffs
                )
            preds = pd.Series()
            # j = 0
            start = datetime.now()
            tracemalloc.start()
            for xi, yi in river_stream.iter_pandas(
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), :-1]
            ):
                forecast = model.predict_one(xi)
                preds = pd.concat([preds, pd.Series(forecast)])
                # j +=1
            diffs = (
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), -1].values - preds
            )
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[int(length)] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
            series_preds = pd.concat([series_preds, preds])
            series_diffs = pd.concat([series_diffs, diffs])
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs


def eval_one_oml_horizon(
    df_eval: pd.DataFrame,
    i: int,
    model: object,
    horizon: int,
    test: pd.DataFrame,
    series_preds: pd.Series,
    series_diffs: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    preds = pd.Series()
    start = datetime.now()
    tracemalloc.start()
    # j = 0
    for xi, yi in river_stream.iter_pandas(
        test.iloc[i * horizon : (i + 1) * horizon, :-1], test.iloc[i * horizon : (i + 1) * horizon, -1]
    ):
        forecast = model.predict_one(xi)
        preds = pd.concat([preds, pd.Series(forecast)])
        # j +=1
    diffs = test.iloc[i * horizon : (i + 1) * horizon, -1].values - preds
    for xi, yi in river_stream.iter_pandas(
        test.iloc[i * horizon : (i + 1) * horizon, :-1], test.iloc[i * horizon : (i + 1) * horizon, -1]
    ):
        model = model.learn_one(xi, yi)
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, (peak / 10**6), time))
    series_preds = pd.concat([series_preds, preds])
    series_diffs = pd.concat([series_diffs, diffs])
    return series_preds, series_diffs, df_eval
