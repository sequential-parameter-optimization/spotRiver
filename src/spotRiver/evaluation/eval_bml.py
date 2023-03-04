import time
import tracemalloc
import numpy as np
import pandas as pd
from river import stream as river_stream
from typing import Optional
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import copy


class ResourceMonitorError(Exception):
    pass


@dataclass
class ResourceUsage:
    name: Optional[str]  # Description of Usage
    time: float  # Measured in seconds
    memory: float  # Measured in bytes

    def __str__(self):
        if self.name is None:
            res = [f"Resource usage for {self.name}:"]
        else:
            res = ["Resource usage:"]
        res.append(f"  Time [s]: {self.time}")
        res.append(f"  Memory [b]: {self.memory}")
        return "\n".join(res)

    def __repr__(self):
        return str(self)


class ResourceMonitor:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.time = None
        self.memory = None
        self.current_memory = None
        self.peak_memory = None
        self._start = None

    def __enter__(self):
        if tracemalloc.is_tracing():
            raise ResourceMonitorError("Already tracing memory usage!")
        tracemalloc.start()
        tracemalloc.reset_peak()
        self._start = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.time = (time.perf_counter_ns() - self._start) / 1.0e9
        _, peak = tracemalloc.get_traced_memory()
        self.memory = peak / (1024 * 1024)
        tracemalloc.stop()

    def result(self):
        if self.time is None or self.memory is None:
            raise ResourceMonitorError("No resources monitored yet.")
        return ResourceUsage(name=self.name, time=self.time, memory=self.memory)


def evaluate_model(
    diff: np.ndarray,
    memory: float,
    time: float,
) -> dict:
    if diff.size == 0:
        res_dict = {
            "RMSE": None,
            "MAE": None,
            "AbsDiff": None,
            "Underestimation": None,
            "Overestimation": None,
            "MaxResidual": None,
            "Memory (MB)": np.round(memory, 6),
            "CompTime (s)": np.round(time, 6),
        }
    else:
        pos_sum = np.sum(diff[diff > 0])
        neg_sum = np.sum(diff[diff < 0])
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))

        res_dict = {
            "RMSE": np.round(rmse, 2),
            "MAE": np.round(mae, 2),
            "AbsDiff": np.round(np.sum(np.abs(diff)), 2),
            "Underestimation": np.round(pos_sum, 2),
            "Overestimation": np.round(np.abs(neg_sum), 2),
            "MaxResidual": np.round(np.max(np.abs(diff)), 2),
            "Memory (MB)": np.round(memory, 6),
            "CompTime (s)": np.round(time, 6),
        }

    return res_dict


def eval_bml_horizon(model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int) -> tuple:
    """Evaluate a model on a batch basis using prediction horizons (mini-batches).

    This function takes a model and two data frames (train and test) as inputs
    and returns two data frames as outputs. The first output contains evaluation
    metrics for each batch of the test data set. The second output contains the
    true and predicted values for each observation in the test data set.

    Parameters
    ----------
    model : object
        The model to be evaluated.
    train : pd.DataFrame
        The initial training data set.
    test : pd.DataFrame
        The testing data set that will be divided into batches of size horizon.
    target_column : str
        The name of the column containing the target variable.
    horizon : int, optional
        The number of steps ahead to forecast.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple of two data frames. The first one contains evaluation metrics for each batch.
        The second one contains the true and predicted values for each observation in the test set.

    Example
    -------
    >>> import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        model = LinearRegression()

        # Create synthetic data for regression with 100 observations, 3 features and one target value
        X_train, y_train = make_regression(n_samples=80, n_features=3, n_targets=1)
        X_test, y_test = make_regression(n_samples=20, n_features=3, n_targets=1)

        # Convert the data into a pandas data frame
        train = pd.DataFrame(X_train, columns=["x1", "x2", "x3"])
        train["y"] = y_train

        test = pd.DataFrame(X_test, columns=["x1", "x2", "x3"])
        test["y"] = y_test

        target_column = "y"
        horizon = 5
        df_eval, df_true = eval_bml_horizion(model , train , test , target_column)
        print (df_eval )
        print (df_true )
    """
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)

    # Initial Training
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    df_eval = pd.DataFrame.from_dict([evaluate_model(np.array([]), rm.memory, rm.time)])

    # Batch Evaluation
    for batch_number, batch_df in test.groupby(np.arange(len(test)) // horizon):
        rm = ResourceMonitor()
        with rm:
            preds = pd.Series(model.predict(batch_df.loc[:, batch_df.columns != target_column]))

        diffs = batch_df[target_column].values - preds
        df_eval.loc[batch_number + 1] = pd.Series(evaluate_model(diffs, rm.memory, rm.time))

        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)

    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def eval_bml_landmark(
    model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int
) -> tuple:
    """Evaluate a model on a landmark basis.

    Args:
        model (object): The model to be evaluated.
        train (pd.DataFrame): The initial training data set.
        test (pd.DataFrame): The testing data set that will be added incrementally to the training set.
        target_column (str): The name of the column containing the target variable.
        horizon (int, optional): The number of steps ahead to forecast.

    Returns:
        tuple: A tuple of two data frames. The first one contains evaluation metrics for each landmark.
               The second one contains the true and predicted values for each observation in the test set.
    Example:
    >>> from sklearn.linear_model import LinearRegression
    >>> model = LinearRegression()
    >>> train = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    >>> test = pd.DataFrame({"x": [4, 5], "y": [8, 10]})
    >>> target_column = "y"
    >>> horizon = 1
    >>> df_eval, df_true = eval_bml_landmark(model, train, test, target_column, horizon)
    >>> print(df_eval)
       MAE   MSE   RMSE   MAPE     Memory      Time
    0  NaN   NaN    NaN    NaN   0.000000  0.000000
    1 -0.0   0.0 -0.000 -0.000   3.906250  0.001001
    >>> print(df_true)
       y Prediction Difference
    0 8        -8         -16
    """
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)

    # Initial Training
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    df_eval = pd.DataFrame.from_dict([evaluate_model(np.array([]), rm.memory, rm.time)])

    # Landmark Evaluation
    for i, new_df in enumerate(gen_sliding_window(test, horizon)):
        train = pd.concat([train, new_df], ignore_index=True)
        rm = ResourceMonitor()
        with rm:
            preds = pd.Series(model.predict(new_df.loc[:, new_df.columns != target_column]))
            model.fit(train.loc[:, train.columns != target_column], train[target_column])

        diffs = new_df[target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, rm.memory, rm.time))

        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)

    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def gen_sliding_window(df, horizon):
    i = 0
    while True:
        subset = df[i * horizon : (i + 1) * horizon]
        if len(subset) == 0:
            break
        elif len(subset) < horizon:
            yield subset
            break
        i += 1
        yield subset


def eval_bml_window(model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int) -> tuple:
    """Evaluate a model on a rolling window basis.

    Args:
        model (object): The model to be evaluated.
        train (pd.DataFrame): The training data set.
        test (pd.DataFrame): The testing data set.
        target_column (str): The name of the column containing the target variable.
        horizon (int, optional): The number of steps ahead to forecast.

    Returns:
        tuple: A tuple of two data frames. The first one contains evaluation metrics for each window.
               The second one contains the true and predicted values for each observation in the test set.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> train = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        >>> test = pd.DataFrame({"x": [4, 5], "y": [8, 10]})
        >>> df_eval, df_true = eval_bml_window(model, train, test, "y", horizon=1)
        >>> print(df_eval)
    """
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    df_all = pd.concat([train, test], ignore_index=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)

    # Window Evaluation
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
    # Initial Training
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    df_eval = pd.DataFrame.from_dict([evaluate_model(np.array([]), rm.memory, rm.time)])

    for i, (w_train, w_test) in enumerate(gen_horizon_shifted_window(df_all, len(train), horizon)):
        rm = ResourceMonitor()
        with rm:
            model.fit(w_train.loc[:, w_train.columns != target_column], w_train[target_column])
            preds = pd.Series(model.predict(w_test.loc[:, w_test.columns != target_column]))

        diffs = w_test[target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, rm.memory, rm.time))

        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)

    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def gen_horizon_shifted_window(df, window_size, horizon):
    i = 0
    while True:
        train_window = df[i * horizon : i * horizon + window_size]
        test_window = df[i * horizon + window_size : (i + 1) * horizon + window_size]
        if len(test_window) == 0:
            break
        elif len(test_window) < horizon:
            yield train_window, test_window
            break
        i += 1
        yield train_window, test_window


def eval_oml_horizon(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    horizon: int,
    oml_grace_period: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate an online machine learning model.

    Parameters
    ----------
    model : object
        The online machine learning model to be evaluated.
    train : pd.DataFrame
        The training data for the model.
    test : pd.DataFrame
        The testing data for the model.
    target_column : str
        The name of the column that contains the target variable.
    horizon : int, optional
        The number of rows to use for each landmark evaluation.
    oml_grace_period: int, optional
        (Short) period used for training the OML before evaluation starts. Can be zero.
        If set to None, than horizon will be used.

    Returns
    -------
    tuple of pd.DataFrame
        The first element is a dataframe with evaluation metrics (RMSE, MAE, memory usage and time elapsed) for each iteration.
        The second element is a dataframe with the true values, predictions and differences for the test data.

    Examples
    -------
    >>> import pandas as pd
        import numpy as np
        from river import linear_model
        from river import preprocessing
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        dataset = datasets.TrumpApproval()
        train = pd.DataFrame(dataset.take(100))
        test = pd.DataFrame(dataset.take(100))
        target_column = "Approve"
        horizon = 10
        df_eval, df_preds = eval_oml_horizon(model, train, test, target_column, horizon, oml_grace_period)
        print(df_eval)
        print(df_preds)
    """
    if oml_grace_period is None:
        oml_grace_period = horizon
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)

    # Initial Training on Train Data
    # For OML, this is performed on a limited subset only (oml_grace_period).
    train_X = train.loc[:, train.columns != target_column]
    train_y = train[target_column]
    train_X = train_X.tail(oml_grace_period)
    train_y = train_y.tail(oml_grace_period)
    rm = ResourceMonitor()
    with rm:
        for xi, yi in river_stream.iter_pandas(train_X, train_y):
            # The following line returns y_pred, which is not used, therefore set to "_":
            _ = model.predict_one(xi)
            # metric = metric.update(yi, y_pred)
            model = model.learn_one(xi, yi)
    df_eval = pd.DataFrame.from_dict([evaluate_model(np.array([]), rm.memory, rm.time)])

    # Test Data Evaluation
    for i, new_df in enumerate(gen_sliding_window(test, horizon)):
        preds = []
        test_X = new_df.loc[:, new_df.columns != target_column]
        test_y = new_df[target_column]
        rm = ResourceMonitor()
        with rm:
            for xi, yi in river_stream.iter_pandas(test_X, test_y):
                pred = model.predict_one(xi)
                preds.append(pred)  # This is falsly measured with the ResourceMonitor
                model = model.learn_one(xi, yi)
        preds = pd.Series(preds)
        diffs = new_df[target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(evaluate_model(diffs, rm.memory, rm.time))
        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)
    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def plot_bml_oml_horizon_metrics(
    df_eval: list[pd.DataFrame] = None,
    df_labels: list = None,
    log_x=False,
    log_y=False,
    cumulative=True,
    **kwargs,
) -> None:
    """Plot metrics for benchmarking machine learning models.

    This function plots three metrics: mean absolute error (MAE), memory usage (MB),
    and computation time (s) for different machine learning models on a given dataset.
    The function takes a list of pandas dataframes as input, each containing the metrics
    for one model. The function also takes an optional list of labels for each model
    and boolean flags to indicate whether to use logarithmic scales for the x-axis
    and y-axis.

    Parameters
    ----------
    df_eval : list[pd.DataFrame], optional
        A list of pandas dataframes containing the metrics for each model.
        Each dataframe should have an index column with the dataset name
        and three columns with the metric names: "MAE", "Memory (MB)", "CompTime (s)".
        If None, no plot is generated. Default is None.

    df_labels : list, optional
        A list of strings containing the labels for each model.
        The length of this list should match the length of df_eval.
        If None, numeric indices are used as labels. Default is None.

    log_x : bool, optional
        A flag indicating whether to use logarithmic scale for the x-axis.
        If True, log scale is used. If False, linear scale is used. Default is False.

    log_y : bool, optional
        A flag indicating whether to use logarithmic scale for the y-axis.
        If True, log scale is used. If False, linear scale is used. Default is False.

    cumulative : bool, optional:
        A flag indicating whether to plot the cumulative average error as it is done in
        `plot_oml_iter_progressive()` and in `river`'s ` evaluate.iter_progressive_val_score()`
        method. Time is shown as a cumulative sum (not averaged). Since memory is calculated
        in a different way than in `river`'s ` evaluate.iter_progressive_val_score()`, the peak
        memory from `_ , peak = tracemalloc.get_traced_memory()` is shown in a non-aggregated way.
        Default is True.

    **kwargs : dict
        Additional keyword arguments passed to plt.plot() function.

    Returns
    -------
    None

    Example
    -------
    >>> d1 = {'MAE': [0.1, 0.2], 'Memory (MB)': [10 , 20], 'CompTime (s)': [1 , 2]}
        d2 = {'MAE': [0.3 , 0.4], 'Memory (MB)': [30 , 40], 'CompTime (s)': [3 , 4]}
        # create dataframes from dictionaries
        df_eval1 = pd.DataFrame(data=d1)
        df_eval2 = pd.DataFrame(data=d2)
        # create a list of dataframes
        df_eval_list = [df_eval1 , df_eval2]
        # plot evaluation metrics for each element of df_eval_list
        plot_bml_oml_metrics(df_eval=df_eval_list)
    """
    # Check if input dataframes are provided
    if df_eval is not None:
        df_list = copy.deepcopy(df_eval)
        # Convert single dataframe input to a list if needed
        if df_list.__class__ != list:
            df_list = [df_list]
        # Define metric names and titles
        metrics = ["MAE", "CompTime (s)", "Memory (MB)"]
        titles = ["Mean Absolute Error", "Computation time (s)", "Memory (MB)"]
        # Create subplots with shared x-axis
        fig, axes = plt.subplots(3, figsize=(16, 5), constrained_layout=True, sharex=True)
        # Loop over each dataframe in input list
        for j, df in enumerate(df_list):
            if cumulative:
                df.MAE = np.cumsum(df.MAE) / range(1, (1 + df.MAE.size))
                df["CompTime (s)"] = np.cumsum(df["CompTime (s)"])  # / range(1, (1 + df["CompTime (s)"].size))
                # df["Memory (MB)"] = np.cumsum(df["Memory (MB)"]) / range(1, (1 + df["Memory (MB)"].size))
            # Loop over each metric
            for i in range(3):
                # Assign label based on input or default value
                if df_labels is None:
                    label = f"{j}"
                else:
                    label = df_labels[j]
                # Plot metric values against dataset names
                axes[i].plot(df.index.values.tolist(), df[metrics[i]].values.tolist(), label=label, **kwargs)
                # Set title and legend
                axes[i].set_title(titles[i])
                axes[i].legend(loc="upper right")
                # Set logarithmic scales if specified
                if log_x:
                    axes[i].set_xscale("log")
                if log_y:
                    axes[i].set_yscale("log")


def plot_bml_oml_horizon_predictions(
    df_true: list[pd.DataFrame] = None,
    df_labels: list = None,
    target_column: str = "Actual",
    log_x=False,
    log_y=False,
    skip_first_n=0,
    **kwargs,
) -> None:
    """Plot actual vs predicted values for machine learning models.

    This function plots the actual values of a target variable (e.g., vibration)
    against the predicted values from different machine learning models on
    a given dataset. The function takes a list of pandas dataframes as input,
    each containing the actual and predicted values for one model. The function
    also takes an optional list of labels for each model and boolean flags
    to indicate whether to use logarithmic scales for the x-axis and y-axis.

    Parameters
    ----------
    df_true : list[pd.DataFrame], optional
        A list of pandas dataframes containing the actual and predicted values
        for each model. Each dataframe should have an index column with the
        dataset name and two columns with the label names: e.g., "Vibration"
        and "Prediction". If None, no plot is generated. Default is None.

    df_labels : list, optional
        A list of strings containing the labels for each model.
        The length of this list should match the length of df_eval.
        If None, numeric indices are used as labels. Default is None.

    target_column : optional
        String containing the target column.
        Default is "Actual".

    log_x : bool, optional
        A flag indicating whether to use logarithmic scale for the x-axis.
        If True, log scale is used. If False, linear scale is used. Default is False.

    log_y : bool, optional
        A flag indicating whether to use logarithmic scale for the y-axis.
        If True, log scale is used. If False, linear scale is used. Default is False.

    skip_first_n: int, optional
        Skip the first n entries in the plot.

    **kwargs : dict
        Additional keyword arguments passed to plt.plot() function.

    Returns
    -------
    None

    Example
    -------
    >>> d3 = {'Vibration': [0.5 , 0.6], 'Prediction': [0.7 , 0.8]}
        # create dataframes from dictionaries
        df_true = pd.DataFrame(data=d3)
        # plot actual vs predicted values from df_true
        plot_bml_oml_horizon_predictions(df_true=df_true)

    """
    if df_true is not None:
        df_plot = copy.deepcopy(df_true)
        if df_plot.__class__ != list:
            df_plot = [df_plot]
        # plot actual vs predicted values
        plt.figure(figsize=(16, 5))
        for j, df in enumerate(df_plot):
            # Assign label based on input or default value
            if df_labels is None:
                label = f"{j}"
            else:
                label = df_labels[j]
            # skip first n values
            df["Prediction"][range(skip_first_n)] = np.nan
            plt.plot(df.index, df["Prediction"], label=label, **kwargs)
        # Plot the actual value only once:
        plt.plot(df_plot[0].index, df_plot[0][target_column], label="Actual", **kwargs)
        plt.title("Actual vs Prediction")
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
        plt.legend()
    plt.show()
