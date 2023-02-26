from river import utils
from river import metrics
from typing import List, Tuple
import matplotlib.pyplot as plt


# Define a function to evaluate a nowcast model using a rolling metric
def eval_nowcast_model(model, dataset, time_interval="month", window_size=12) -> Tuple:
    """
    Evaluates a time series model using a rolling mean absolute error metric.

    Parameters:
        model: A predictor object (river.time_series) that implements the forecast and learn_one methods.
        dataset: A dataset object that contains the time series data.
        time_interval: The name of the attribute that contains the date information in the dataset.
        window_size: The number of observations to use for calculating the rolling metric.

    Returns:
        A tuple of four lists:
            - dates: The dates corresponding to each observation in the dataset.
            - metric: A rolling metric object that contains the mean absolute error values.
            - y_trues: The true values of the target variable in the dataset.
            - y_preds: The predicted values of the target variable by the model.

    Examples:
        >>> from river import compose
            from river import linear_model
            from river import preprocessing, datasets, utils, metrics
            import matplotlib.pyplot as plt
            from spotRiver.utils.features import get_ordinal_date
            from spotRiver.evaluation.eval_nowcast import eval_nowcast_model, plot_nowcast_model

            model = compose.Pipeline(
                ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
                ('scale', preprocessing.StandardScaler()),
                ('lin_reg', linear_model.LinearRegression())
            )
            dataset = datasets.AirlinePassengers()
            dates, metric, y_trues, y_preds = eval_nowcast_model(model, dataset=dataset)
            plot_nowcast_model(dates, metric, y_trues, y_preds)
    """
    metric = utils.Rolling(obj=metrics.MAE(), window_size=window_size)
    dates = []
    y_trues = []
    y_preds = []
    for x, y in dataset:
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        # Update the error metric
        metric.update(y, y_pred)
        # Store the true value and the prediction
        dates.append(x[time_interval])
        y_trues.append(y)
        y_preds.append(y_pred)
    return dates, metric, y_trues, y_preds


def plot_nowcast_model(
    dates: List[str], metric: utils.Rolling, y_trues: List[float], y_preds: List[float], range: List[int] = None
) -> None:
    """
    Plots the true values and the predictions of a nowcast model along with a rolling metric.

    Parameters:
        dates: A list of strings that contains the dates corresponding to each observation.
        metric: A rolling metric object that contains the mean absolute error values.
        y_trues: A list of floats that contains the true values of the target variable.
        y_preds: A list of floats that contains the predicted values of the target variable.
        range: A list of 2 int that specify the subset.

    Returns:
        None. Displays a matplotlib figure with two lines and a title.
    """
    # Create a figure and an axis object
    fig, ax = plt.subplots(figsize=(10, 6))
    # Add grid lines to the plot
    ax.grid(alpha=0.75)
    if range is not None:
        dates = dates[range[0] : range[1]]
        y_preds = y_preds[range[0] : range[1]]
        y_trues = y_trues[range[0] : range[1]]
    # Plot the true values and the predictions with different colors and labels
    ax.plot(dates, y_trues, lw=3, color="#2ecc71", alpha=0.8, label="Ground truth")
    ax.plot(dates, y_preds, lw=3, color="#e74c3c", alpha=0.8, label="Prediction")
    # Add a legend to show the labels
    ax.legend()
    # Set the title of the plot to be the rolling metric
    ax.set_title(metric)
