import time
import tracemalloc
import numpy as np
import pandas as pd
from river import stream as river_stream
from typing import Optional
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt


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
        self._start = None

    def __enter__(self):
        if tracemalloc.is_tracing():
            raise ResourceMonitorError("Already tracing memory usage!")
        tracemalloc.start()
        tracemalloc.reset_peak()
        self._start = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.time = (time.perf_counter_ns() - self._start) / 1.0e9
        self.memory = np.round(tracemalloc.get_traced_memory()[1] / 1.0e6, 4)
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
            "Memory (MB)": np.round(memory, 4),
            "CompTime (s)": np.round(time, 4),
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
            "Memory (MB)": np.round(memory, 4),
            "CompTime (s)": np.round(time, 4),
        }

    return res_dict


def eval_bml(model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int = 1) -> tuple:
    """Evaluate a model on a batch basis.

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
        The number of steps ahead to forecast. Defaults to 1.

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

        # Create a linear regression model
        model = LinearRegression()

        # Create synthetic data for regression with 100 observations, 3 features and one target value
        X_train, y_train = make_regression(n_samples=80, n_features=3, n_targets=1)
        X_test, y_test = make_regression(n_samples=20, n_features=3, n_targets=1)

        # Convert the data into a pandas data frame
        train = pd.DataFrame(X_train, columns=["x1", "x2", "x3"])
        train["y"] = y_train

        test = pd.DataFrame(X_test, columns=["x1", "x2", "x3"])
        test["y"] = y_test

        # Set the name of the target variable
        target_column = "y"

         # Set the horizon to 5
         horizon = 5

         # Evaluate the model on a batch basis
         df_eval , df_true = eval_bml (model , train , test , target_column )

         # Print the results
         print (df_eval )
         print (df_true )
    """
    series_preds = pd.Series([])
    series_diffs = pd.Series([])

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
    model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int = 1
) -> tuple:
    """Evaluate a model on a landmark basis.

    Args:
        model (object): The model to be evaluated.
        train (pd.DataFrame): The initial training data set.
        test (pd.DataFrame): The testing data set that will be added incrementally to the training set.
        target_column (str): The name of the column containing the target variable.
        horizon (int, optional): The number of steps ahead to forecast. Defaults to 1.

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
    series_preds = pd.Series([])
    series_diffs = pd.Series([])

    # Initial Training
    rm = ResourceMonitor()
    with rm:
        model.fit(train.loc[:, train.columns != target_column], train[target_column])
    df_eval = pd.DataFrame.from_dict([evaluate_model(np.array([]), rm.memory, rm.time)])

    # Landmark Evaluation
    for i, new_df in enumerate(landmark_gen(test, horizon)):
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


def landmark_gen(df, horizon):
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


def eval_bml_window(
    model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int = 1
) -> tuple:
    """Evaluate a model on a rolling window basis.

    Args:
        model (object): The model to be evaluated.
        train (pd.DataFrame): The training data set.
        test (pd.DataFrame): The testing data set.
        target_column (str): The name of the column containing the target variable.
        horizon (int, optional): The number of steps ahead to forecast. Defaults to 1.

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
    df_all = pd.concat([train, test], ignore_index=True)
    series_preds = pd.Series([])
    series_diffs = pd.Series([])

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
    for i, (w_train, w_test) in enumerate(window_gen(df_all, len(train), horizon)):
        rm = ResourceMonitor()
        with rm:
            model.fit(w_train.loc[:, w_train.columns != target_column], w_train[target_column])
            preds = pd.Series(model.predict(w_test.loc[:, w_test.columns != target_column]))

        diffs = w_test[target_column].values - preds
        df_eval.loc[i] = pd.Series(evaluate_model(diffs, rm.memory, rm.time))

        series_preds = pd.concat([series_preds, preds], ignore_index=True)
        series_diffs = pd.concat([series_diffs, diffs], ignore_index=True)

    df_true = pd.DataFrame(test[target_column])
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true


def window_gen(df, window_size, horizon):
    i = 0
    while True:
        train_window = df[i * horizon : window_size + (i * horizon)]
        test_window = df[window_size + (i * horizon) : window_size + ((i + 1) * horizon)]
        if len(test_window) == 0:
            break
        elif len(test_window) < horizon:
            yield train_window, test_window
            break
        i += 1
        yield train_window, test_window


def eval_oml_landmark(
    model: object, train: pd.DataFrame, test: pd.DataFrame, target_column: str, horizon: int = 1
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
        The number of rows to use for each landmark evaluation. Default is 1.

    Returns
    -------
    tuple of pd.DataFrame
        The first element is a dataframe with evaluation metrics (RMSE, MAE, memory usage and time elapsed) for each iteration.
        The second element is a dataframe with the true values, predictions and differences for the test data.

    Examples
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from river import linear_model
    >>> from river import preprocessing
    >>> model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    >>> dataset = datasets.TrumpApproval()
    >>> train = pd.DataFrame(dataset.take(100))
    >>> test = pd.DataFrame(dataset.take(100))
    >>> target_column = "Approve"
    >>> horizon = 10
    >>> df_eval, df_preds = eval_oml_landmark(model, train, test, target_column, horizon)
    >>> print(df_eval)
    >>> print(df_preds)
    """
    series_preds = pd.Series([])
    series_diffs = pd.Series([])

    # Initial Training
    rm = ResourceMonitor()
    with rm:
        for xi, yi in river_stream.iter_pandas(train.loc[:, train.columns != target_column], train[target_column]):
            model = model.learn_one(xi, yi)
    df_eval = pd.DataFrame.from_dict([evaluate_model(np.array([]), rm.memory, rm.time)])

    # Landmark Evaluation
    for i, new_df in enumerate(landmark_gen(test, horizon)):
        train = pd.concat([train, new_df], ignore_index=True)
        preds = []
        rm = ResourceMonitor()
        with rm:
            for xi, _ in river_stream.iter_pandas(
                new_df.loc[:, new_df.columns != target_column], new_df[target_column]
            ):
                pred = model.predict_one(xi)
                preds.append(pred)  # This is falsly measured with the ResourceMonitor
            for xi, yi in river_stream.iter_pandas(
                new_df.loc[:, new_df.columns != target_column], new_df[target_column]
            ):
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


def plot_bml_oml_results(
    df_eval: list[pd.DataFrame] = None,
    metric: bool = False,
    df_true: list[pd.DataFrame] = None,
    real_vs_predict: bool = False,
    actual_val_name="Vibration",
    predicted_val_name="Prediction",
    log_x=False,
    log_y=False,
    **kwargs,
) -> None:
    """Plots evaluation results or actual vs predicted values.

    This function creates plots to visualize evaluation results or actual vs predicted values.
    The function does not return anything, it only generates and shows the plots.

    Args:
        df_eval (list[pd.DataFrame], optional): A list of dataframes containing evaluation results. Each dataframe should have columns named "MAE", "Memory (MB)", and "CompTime (s)". Defaults to None.
        metric (bool, optional): If True, plot evaluation metrics for each element of df_eval. Defaults to False.
        df_true (list[pd.DataFrame], optional): A lisrt of dataframes containing actual and predicted values. The dataframes should have columns named by actual_val_name and predicted_val_name arguments. Defaults to None.
        real_vs_predict (bool, optional): If True, plot actual vs predicted values from df_true. Defaults to False.
        actual_val_name (str, optional): The name of the column that contains the actual values in df_true. Defaults to "Vibration".
        predicted_val_name (str, optional): The name of the column that contains the predicted values in df_true. Defaults to "Prediction".
        log_x (bool): Plot log x-axis.
        log_y (bool): Plot log y-axis.
        **kwargs: Additional keyword arguments that are passed to the plt.plot() function calls.

    Returns:
        None

    Example:
        >>> # create some sample data
            d1 = {'MAE': [0.1, 0.2], 'Memory (MB)': [10 , 20], 'CompTime (s)': [1 , 2]}
            d2 = {'MAE': [0.3 , 0.4], 'Memory (MB)': [30 , 40], 'CompTime (s)': [3 , 4]}
            d3 = {'Vibration': [0.5 , 0.6], 'Prediction': [0.7 , 0.8]}

            # create dataframes from dictionaries
            df_eval1 = pd.DataFrame(data=d1)
            df_eval2 = pd.DataFrame(data=d2)
            df_true = pd.DataFrame(data=d3)

            # create a list of dataframes
            df_eval_list = [df_eval1 , df_eval2]

            # call the function with different arguments

            # plot evaluation metrics for each element of df_eval_list
            plot_bml_oml_results(df_eval=df_eval_list , metric=True)

            # plot actual vs predicted values from df_true
            plot_bml_oml_results(df_true=df_true , real_vs_predict=True)
    """
    if df_eval is not None and metric:
        # create a list of metrics and titles
        metrics = ["MAE", "Memory (MB)", "CompTime (s)"]
        titles = ["Mean Absolute Error", "Memory (MB)", "Computation time (s)"]
        # create a 3 subplot figure to plot evaluation metrics for each element
        fig, axes = plt.subplots(3, figsize=(16, 5), constrained_layout=True, sharex=True)
        # use a loop to iterate over each element of df_eval
        for j, df in enumerate(df_eval):
            # use a loop to plot each metric on a different axis
            for i in range(3):
                axes[i].plot(df.index, df[metrics[i]], label=str(j), **kwargs)
                axes[i].set_title(titles[i])
                axes[i].legend(loc="upper right")
                if log_x:
                    axes[i].set_xscale("log")
                if log_y:
                    axes[i].set_yscale("log")
            # add a suptitle for each element
            fig.suptitle(f"Element {j+1} of df_eval")
    if df_true is not None and real_vs_predict:
        # plot actual vs predicted values
        plt.figure(figsize=(16, 5))
        # Plot the actual value only once:
        plt.plot(df_true[0].index, df_true[0][actual_val_name], label="Actual", **kwargs)
        for j, df in enumerate(df_true):
            plt.plot(df.index, df[predicted_val_name], label="Prediction", **kwargs)
        plt.title("Actual vs Prediction")
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
        plt.legend()
    plt.show()
