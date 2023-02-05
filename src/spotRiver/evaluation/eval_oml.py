import matplotlib.pyplot as plt
from river.evaluate import iter_progressive_val_score
from spotPython.utils.progress import progress_bar
from numpy import median
from numpy import zeros


def eval_oml_iter_progressive(dataset, metric, models, step=100, verbose=False):
    """Evaluate OML Models

    Args:
        dataset:
        metric:
        models:
        step (int): Iteration number at which to yield results.
            This only takes into account the predictions, and not the training steps.
        verbose:

    Reference:
        https://riverml.xyz/0.15.0/recipes/on-hoeffding-trees/
    """
    metric_name = metric.__class__.__name__
    dataset = list(dataset)
    n_steps = len(dataset)
    result = {}
    for model_name, model in models.items():
        result_i = {"step": [], "error": [], "r_time": [], "memory": []}
        for checkpoint in iter_progressive_val_score(
            dataset, model, metric, measure_time=True, measure_memory=True, step=step
        ):
            if verbose:
                progress_bar(checkpoint["Step"] / n_steps, message="Eval iter_prog_val_score:")
            result_i["step"].append(checkpoint["Step"])
            result_i["error"].append(checkpoint[metric_name].get())
            # Convert timedelta object into seconds
            result_i["r_time"].append(checkpoint["Time"].total_seconds())
            # Make sure the memory measurements are in MB
            raw_memory = checkpoint["Memory"]
            result_i["memory"].append(raw_memory * 2**-20)
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


def fun_eval_oml_iter_progressive(result, metric=None):
    """
    Wrapper for eval_oml_iter_progressive. Returns one function value,
    e.g., for objective functions.

    Args:
        result (_type_): _description_
        metric (_type_, optional): _description_. Defaults to None.
    """
    if metric is None:
        metric = median
    model_names = list(result.keys())
    n = len(model_names)
    y = zeros([n])
    for i in range(n):
        y[i] = metric(result[model_names[i]]["error"])
    return y
