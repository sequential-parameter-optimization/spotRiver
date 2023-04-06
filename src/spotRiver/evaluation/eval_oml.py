import matplotlib.pyplot as plt
from river.evaluate import iter_progressive_val_score
from spotPython.utils.progress import progress_bar
from numpy import mean
from numpy import zeros
from numpy import array


def eval_oml_iter_progressive(dataset, metric, models, step=100, weight_coeff=0.0, log_level=50):
    """Evaluate OML Models on Streaming Data

    This function evaluates one or more OML models on a streaming dataset. The evaluation
    is done iteratively, and the models are tested every `step` iterations. The results
    are returned as a dictionary of metrics and their values.

    Args:
        dataset (list or river.Stream): A list of river.Stream objects containing the
            streaming data to be evaluated. If a single river.Stream object is provided,
            it is automatically converted to a list.
        metric (river.metrics.base.MultiClassMetric or river.metrics.base.RegressionMetric):
            The metric to be used for evaluation.
        models (dict): A dictionary of OML models to be evaluated. The keys are the names
            of the models, and the values are the model objects.
        step (int): Iteration number at which to yield results. This only takes into account
            the predictions, and not the training steps. Defaults to 100.
        weight_coeff (float): Results are multiplied by (step/n_steps)**weight_coeff,
            where n_steps is the total number of iterations. Results from the beginning have
            a lower weight than results from the end if weight_coeff > 1. If weight_coeff == 0,
            then results are multiplied by 1 and every result has an equal weight. Defaults to 0.0.
        log_level (int): The level of logging to use. 0 = no logging, 50 = print only important
            information. Defaults to 50.

    Returns:
        dict: A dictionary containing the evaluation results. The keys are the names of the
            models, and the values are dictionaries with the following keys:
            - "step": A list of iteration numbers at which the model was evaluated.
            - "error": A list of the weighted errors for each iteration.
            - "r_time": A list of the weighted running times for each iteration.
            - "memory": A list of the weighted memory usages for each iteration.
            - "metric_name": The name of the metric used for evaluation.

    Reference:
        https://riverml.xyz/0.15.0/recipes/on-hoeffding-trees/
    """
    metric_name = metric.__class__.__name__
    # Convert dataset to a list if needed
    if dataset.__class__ != list:
        dataset = [dataset]
    n_steps = len(dataset)
    result = {}
    for model_name, model in models.items():
        result_i = {"step": [], "error": [], "r_time": [], "memory": []}
        for checkpoint in iter_progressive_val_score(
            dataset, model, metric, measure_time=True, measure_memory=True, step=step
        ):
            if log_level <= 20:
                progress_bar(checkpoint["Step"] / n_steps, message="Eval iter_prog_val_score:")
            w = (checkpoint["Step"] / n_steps) ** weight_coeff
            result_i["step"].append(checkpoint["Step"])
            result_i["error"].append(w * checkpoint[metric_name].get())
            # Convert timedelta object into seconds
            result_i["r_time"].append(w * checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory = checkpoint["Memory"]
            result_i["memory"].append(w * raw_memory * 2**-20)
        result_i["metric_name"] = metric_name
        result[model_name] = result_i
    return result


def plot_oml_iter_progressive(result, log_x=False, log_y=False, figsize=None, filename=None):
    """Plot evaluation of OML models.

    Args:
        result (dict): A dictionary of evaluation results, as returned by eval_oml_iter_progressive.
        log_x (bool, optional): If True, the x-axis is set to log scale. Defaults to False.
        log_y (bool, optional): If True, the y-axis is set to log scale. Defaults to False.
        figsize (tuple, optional): The size of the figure. Defaults to None, in which case
            the default figure size `(10, 5)` is used.
        filename (str, optional): The name of the file to save the plot to. If None, the plot
            is not saved. Defaults to None.

    Reference:
        https://riverml.xyz/0.15.0/recipes/on-hoeffding-trees/
    """
    if figsize is None:
        figsize = (10, 5)
    fig, ax = plt.subplots(figsize=figsize, nrows=3, dpi=300)
    for model_name, model in result.items():
        ax[0].plot(model["step"], model["error"], label=model_name)
        ax[1].plot(model["step"], model["r_time"], label=model_name)
        ax[2].plot(model["step"], model["memory"], label=model_name)

    ax[0].set_ylabel(model["metric_name"])
    ax[1].set_ylabel("Time (seconds)")
    ax[2].set_ylabel("Memory (MB)")
    ax[2].set_xlabel("Instances")

    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    if log_y:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[2].set_yscale("log")

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.close()
    if filename is not None:
        fig.savefig(filename, dpi=300)
    return fig


def fun_eval_oml_iter_progressive(result, metric=None, weights=None):
    """Wrapper function for eval_oml_iter_progressive, returning a single function value.

    Args:
        result (dict): A dictionary of evaluation results, as returned by eval_oml_iter_progressive.
        metric (function, optional): The metric function to use for computing the function value.
            Defaults to None, in which case the mean function is used.
        weights (numpy.array, optional): An array of weights for error, r_time, and memory.
            If None, the weights are set to [1, 0, 0], which considers only the error.
            Defaults to None.

    Returns:
        numpy.array: An array of function values, one for each model in the evaluation results.

    Raises:
        ValueError: If the weights array is not of length 3.
    """
    if metric is None:
        metric = mean
    if weights is None:
        weights = array([1, 0, 0])
    if len(weights) != 3:
        raise ValueError("The weights array must be of length 3.")
    model_names = list(result.keys())
    n = len(model_names)
    y = zeros([n])
    for i in range(n):
        y[i] = (
            weights[0] * metric(result[model_names[i]]["error"])
            + weights[1] * metric(result[model_names[i]]["r_time"])
            + weights[2] * metric(result[model_names[i]]["memory"])
        )
    return y
