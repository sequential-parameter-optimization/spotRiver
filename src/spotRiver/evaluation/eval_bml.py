import time
import tracemalloc
import numpy as np
import pandas as pd
from river import stream as river_stream
from typing import Optional
from dataclasses import dataclass
from typing import Tuple, Generator
import matplotlib.pyplot as plt
import copy


class ResourceMonitorError(Exception):
    pass


@dataclass
class ResourceUsage:
    name: Optional[str]  # Description of Usage
    r_time: float  # Measured in seconds
    memory: float  # Measured in bytes

    def __str__(self):
        if self.name is None:
            res = [f"Resource usage for {self.name}:"]
        else:
            res = ["Resource usage:"]
        res.append(f"  Time [s]: {self.r_time}")
        res.append(f"  Memory [b]: {self.memory}")
        return "\n".join(res)

    def __repr__(self):
        return str(self)


class ResourceMonitor:
    """
    A class to monitor resource usage such as memory and time.

    Attributes
    ----------
    name : str
        The name of the resource monitor.
    r_time : float
        The time taken for the monitored code to execute in seconds.
    memory : float
        The peak memory usage during the execution of the monitored code in MB.

    Methods
    -------
    result():
        Returns a ResourceUsage object containing the name, time and memory usage of the monitored code.

    Example
    -------
        >>> rm = ResourceMonitor()
            with rm:
                x = 10 ** 6
            print(rm.result())
            # Output:
            #   Resource usage for None:
            #   Time [s]: 2.917e-06
            #   Memory [b]: 8.7738037109375e-05
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.r_time = None
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
        self.r_time = (time.perf_counter_ns() - self._start) / 1.0e9
        _, peak = tracemalloc.get_traced_memory()
        self.memory = peak / (1024 * 1024)
        tracemalloc.stop()

    def result(self):
        if self.r_time is None or self.memory is None:
            raise ResourceMonitorError("No resources monitored yet.")
        return ResourceUsage(name=self.name, r_time=self.r_time, memory=self.memory)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, memory: float, r_time: float, metric) -> dict:
    """
    Evaluate a model's performance based on its predictions and ground truth values.

    Parameters
    ----------
    y_true: Ground truth values as a numpy array.
    y_pred: Predicted values as a numpy array.
    memory: Memory usage in MB.
    r_time: Computation time in seconds.
    metric: A function that takes in two arguments (y_true and y_pred) and returns a score.

    Returns
    -------
    A dictionary containing the evaluation results including the metric score, memory usage and computation time.

    Example
    ------
    >>> from sklearn.metrics import mean_squared_error
        import numpy as np
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        memory = 100
        r_time = 0.5
        result = evaluate_model(y_true=y_true,
                                y_pred=y_pred,
                                memory=memory,
                                r_time=r_time,
                                metric=mean_squared_error)
        print(result)

        # Output:
        # {'Metric': 0.00666666666666671, 'Memory (MB)': 100, 'CompTime (s)': 0.5}
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same size")
    if (len(y_true) == 0) or (len(y_pred) == 0):
        res_dict = {
            "Metric": None,
            "Memory (MB)": memory,
            "CompTime (s)": r_time,
        }
        return res_dict
    score = metric(y_true, y_pred)
    res_dict = {"Metric": score, "Memory (MB)": memory, "CompTime (s)": r_time}
    return res_dict


def eval_bml_horizon(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
) -> tuple:
    """
    Evaluate a machine learning model on a rolling horizon basis.

    This function evaluates a machine learning model on a test dataset by making
    predictions on batches of size `horizon` from the test dataset. The model is
    first fit on the training dataset. The evaluation results are returned as a
    dataframe along with a dataframe containing the true values, predictions and
    differences for each observation in the test dataset.

    Parameters
    ----------
    model : object
        A machine learning model object that has `fit` and `predict` methods.

    train : pd.DataFrame
        A pandas DataFrame containing the training data. The target column should be included in this dataframe.

     test : pd.DataFrame
         A pandas DataFrame containing the test data. The target column should be included in this dataframe.

     target_column : str
         The name of the target column in the train and test dataframes.

     horizon : int
         The size of each batch from the test dataframe to make predictions on.

     include_remainder : bool, optional (default=True)
         Whether to include remainder rows from the test dataframe when making predictions.
         If False, remainder rows will be removed before making predictions.

     metric : object
         An evaluation metric object that has an `evaluate` method.
         This metric will be used to evaluate the model's performance on each batch from the test dataframe.

     Returns
     -------
     tuple of pd.DataFrame
         A tuple containing two pandas DataFrames. The first dataframe contains evaluation results for
         each batch from the test dataframe along with initial resource usage from fitting the model
         on the training data. The second dataframe contains true values, predictions and differences
         for each observation in the test dataset.

     Example
     -------
        >>> from sklearn.datasets import make_classification
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score
            # Generate synthetic data for classification
            X_train, y_train = make_classification(n_samples=1000)
            X_test, y_test = make_classification(n_samples=1000)
            # Convert to DataFrames and add target column
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train
            test_df = pd.DataFrame(X_test)
            test_df['target'] = y_test
            # Initialize Decision Tree Classifier and evaluate using eval_bml_horizon function
            dtc = DecisionTreeClassifier()
            eval_results_df, true_preds_diffs_df = eval_bml_horizon(dtc, train_df,test_df,'target', 10, True, accuracy_score)
            # Print first 5 rows of evaluation results DataFrame
            print(eval_results_df.head())
            # Print first 5 rows of true values/predictions/differences DataFrame
            print(true_preds_diffs_df.head())
    """
    # Check if metric is None or null and raise ValueError if it is
    if metric is None:
        raise ValueError("The 'metric' parameter must not be None or null.")
    # Reset index of train and test dataframes
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    # Initialize lists for predictions and differences
    preds_list = []
    diffs_list = []
    # Fit the model on the training data
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    # Evaluate the model on empty arrays to get initial resource usage
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )
    # If include_remainder is False, remove remainder rows from test dataframe
    if include_remainder is False:
        remainder = len(test) % horizon
        if remainder > 0:
            test = test[:-remainder]
    # Evaluate the model on batches of size horizon from the test dataframe
    for batch_number, batch_df in test.groupby(np.arange(len(test)) // horizon):
        rm = ResourceMonitor()
        with rm:
            preds = model.predict(batch_df.loc[:, batch_df.columns != target_column])
        diffs = batch_df[target_column].values - preds
        df_eval.loc[batch_number + 1] = pd.Series(
            evaluate_model(
                y_true=batch_df[target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )
        # Append predictions and differences to their respective lists
        preds_list.append(preds)
        diffs_list.append(diffs)
    # Concatenate predictions and differences lists into series
    series_preds = pd.Series(np.concatenate(preds_list))
    series_diffs = pd.Series(np.concatenate(diffs_list))
    # Create a dataframe with true values and add columns for predictions and differences
    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def eval_bml_landmark(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
) -> tuple:
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    series_preds = pd.Series(dtype=float)
    series_diffs = pd.Series(dtype=float)
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]
    # Landmark Evaluation
    for i, new_df in enumerate(gen_sliding_window(test, horizon)):
        train = pd.concat([train, new_df], ignore_index=True)
        rm = ResourceMonitor()
        with rm:
            preds = pd.Series(model.predict(new_df.loc[:, new_df.columns != target_column]))
            model.fit(train.loc[:, train.columns != target_column], train[target_column])

        diffs = new_df[target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(
            evaluate_model(
                y_true=new_df[target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )
        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)
    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def gen_sliding_window(
    df: pd.DataFrame, horizon: int, include_remainder: bool = True
) -> Generator[pd.DataFrame, None, None]:
    """Generates sliding windows of a given size from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        horizon (int): The size of the sliding window.
        include_remainder (bool): Whether to include the remainder of the DataFrame if its length is not divisible by the horizon. Defaults to False.

    Yields:
        pd.DataFrame: A sliding window of the input DataFrame.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> for window in gen_sliding_window(df, 2):
        ...     print(window)
           A  B
        0  1  4
        1  2  5
           A  B
        2  3  6
    """
    i = 0
    while True:
        subset = df[i * horizon : (i + 1) * horizon]
        if len(subset) == 0:
            break
        elif len(subset) < horizon:
            if include_remainder:
                yield subset
            break
        i += 1
        yield subset


def eval_bml_window(
    model: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    horizon: int,
    include_remainder: bool = True,
    metric: object = None,
) -> tuple:
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
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]
    for i, (w_train, w_test) in enumerate(gen_horizon_shifted_window(df_all, len(train), horizon)):
        rm = ResourceMonitor()
        with rm:
            model.fit(w_train.loc[:, w_train.columns != target_column], w_train[target_column])
            preds = pd.Series(model.predict(w_test.loc[:, w_test.columns != target_column]))

        diffs = w_test[target_column].values - preds
        df_eval.loc[i + 1] = pd.Series(
            evaluate_model(
                y_true=w_test[target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )

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
    include_remainder: bool = True,
    metric: object = None,
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
    metric: object
        Metric, e.g., from sklearn.metrics: mean_squared_error
    oml_grace_period: int, optional
        (Short) period used for training the OML before evaluation starts. Can be zero.
        If set to None, horizon will be used.

    Returns
    -------
    tuple of pd.DataFrame
        The first element is a dataframe with evaluation metrics (RMSE, MAE, ACC, memory usage and time elapsed) for each iteration.
        The second element is a dataframe with the true values, predictions and differences for the test data.

    Examples
    -------
    >>> import pandas as pd
        import numpy as np
        from river import linear_model
        from river import preprocessing
        from sklearn.metrics import mean_squared_error
        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        dataset = datasets.TrumpApproval()
        train = pd.DataFrame(dataset.take(100))
        test = pd.DataFrame(dataset.take(100))
        target_column = "Approve"
        horizon = 10
        metric = mean_squared_error
        df_eval, df_preds = eval_oml_horizon(model, train, test, target_column, horizon, oml_grace_period, metric=metric)
        print(df_eval)
        print(df_preds)
    """
    # Check if metric is None or null and raise ValueError if it is
    if metric is None:
        raise ValueError("The 'metric' parameter must not be None or null.")
    if oml_grace_period is None:
        oml_grace_period = horizon
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if include_remainder is False:
        rem = len(test) % horizon
        if rem > 0:
            test = test[:-rem]
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
    df_eval = pd.DataFrame.from_dict(
        [evaluate_model(y_true=np.array([]), y_pred=np.array([]), memory=rm.memory, r_time=rm.r_time, metric=metric)]
    )

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
        df_eval.loc[i + 1] = pd.Series(
            evaluate_model(
                y_true=new_df[target_column],
                y_pred=preds,
                memory=rm.memory,
                r_time=rm.r_time,
                metric=metric,
            )
        )
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
    grid=True,
    fig_width=16,
    fig_height=5,
    metric=None,
    filename=None,
    **kwargs,
) -> None:
    """Plot metrics for benchmarking machine learning models.

    This function plots three metrics: mean absolute error (MAE)/ Accuracy, memory usage (MB),
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
        and three columns with the metric names: "MAE"/"ACC", "Memory (MB)", "CompTime (s)".
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

    grid: bool, optional
        A flag indicating whether to plot a grid.
        If True, grid is shown. Default is True.

    fig_width: float, optional. Default: 16
        width in inches.

    fig_height: float, optional. Default: 5
        height in inches.

    filename: str, optional. Default: None.
        If not None, the plot is saved to the specified file.

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
    # Check if metric is None or null and raise ValueError if it is
    if metric is None:
        raise ValueError("The 'metric' parameter must not be None or null.")
    # Check if input dataframes are provided
    if df_eval is not None:
        df_list = copy.deepcopy(df_eval)
        # Convert single dataframe input to a list if needed
        if df_list.__class__ != list:
            df_list = [df_list]
        # Define metric names and titles
        metric_name = metric.__name__
        metrics = ["Metric", "CompTime (s)", "Memory (MB)"]
        titles = [metric_name, "Computation time (s)", "Memory (MB)"]
        # Create subplots with shared x-axis
        fig, axes = plt.subplots(3, figsize=(fig_width, fig_height), constrained_layout=True, sharex=True)
        # Loop over each dataframe in input list
        for j, df in enumerate(df_list):
            if cumulative:
                # df.MAE = np.cumsum(df.MAE) / range(1, (1 + df.MAE.size))
                df["Metric"] = np.cumsum(df["Metric"]) / range(1, (1 + df["Metric"].size))
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
                axes[i].grid(grid)
                # Set logarithmic scales if specified
                if log_x:
                    axes[i].set_xscale("log")
                if log_y:
                    axes[i].set_yscale("log")
        if filename is not None:
            plt.savefig(filename)


def plot_bml_oml_horizon_predictions(
    df_true: list[pd.DataFrame] = None,
    df_labels: list = None,
    target_column: str = "Actual",
    log_x=False,
    log_y=False,
    skip_first_n=0,
    grid=True,
    fig_width=16,
    fig_height=5,
    filename=None,
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

    grid: bool, optional
        A flag indicating whether to plot a grid.
        If True, grid is shown. Default is True.

    fig_width: float, optional. Default: 16
        width in inches.

    fig_height: float, optional. Default: 5
        height in inches.

    filename: str, optional. Default: None
        If not None, save the plot to a file.

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
        plt.figure(figsize=(fig_width, fig_height))
        for j, df in enumerate(df_plot):
            if df_labels is None:
                label = f"{j}"
            else:
                label = df_labels[j]
            df.loc[: skip_first_n - 1, "Prediction"] = np.nan
            plt.plot(df.index, df["Prediction"], label=label, **kwargs)
        plt.plot(df_plot[0].index, df_plot[0][target_column], label="Actual", color="black", **kwargs)
        plt.title("Actual vs Prediction")
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
        plt.grid(grid)
        plt.legend()
        if filename is not None:
            plt.savefig(filename)
    plt.show()
