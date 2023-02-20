import matplotlib.pyplot as plt
from river.evaluate import iter_progressive_val_score
from spotPython.utils.progress import progress_bar
from numpy import mean
from numpy import zeros
from numpy import array


def eval_oml_iter_progressive(dataset, metric, models, step=100, weight_coeff=0.0, log_level=50):
    """Evaluate OML Models

    Args:
        dataset:
        metric:
        models:
        step (int): Iteration number at which to yield results.
            This only takes into account the predictions, and not the training steps.
        weight_coeff (float): results are multiplied by (step/n_steps)**weight_coeff,
            results from the beginning have a lower weight than results from the end,
            if weight_coeff > 1. If weight_coeff == 0, than results are multiplied by 1
            and every result has an equal weight.
        log_level:

    Reference:
        https://riverml.xyz/0.15.0/recipes/on-hoeffding-trees/
    """
    metric_name = metric.__class__.__name__
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


def plot_oml_iter_progressive(result, log_y=False):
    """Plot evaluation of OML models.

    Args:
        result (dict):

    Reference:
        https://riverml.xyz/0.15.0/recipes/on-hoeffding-trees/
    """
    fig, ax = plt.subplots(figsize=(10, 5), nrows=3, dpi=300)
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

    return fig


def fun_eval_oml_iter_progressive(result, metric=None, weights=None):
    """
    Wrapper for eval_oml_iter_progressive. Returns one function value,
    e.g., for objective functions.

    Args:
        result (_type_): _description_
        metric (_type_, optional): _description_. Defaults to None.
        weights (numpy.array): Weights for error, r_time, and memory. None is [1,0,0],
            which considers error only. Defaults to None.
    """
    if metric is None:
        metric = mean
    if weights is None:
        weights = array([1, 0, 0])
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
