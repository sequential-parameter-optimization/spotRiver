import matplotlib.pyplot as plt
from river.evaluate import iter_progressive_val_score


def eval_oml_iter_progressive(dataset, metric, models, step=100, verbose=False):
    """ Evaluate OML Models

    Args:
        dataset:
        metric:
        models:

    Reference:
        https://riverml.xyz/0.15.0/recipes/on-hoeffding-trees/
    """
    metric_name = metric.__class__.__name__
    dataset = list(dataset)
    result = {}
    for model_name, model in models.items():
        result_i = {"step": [], "error": [], "r_time": [], "memory": []}
        for checkpoint in iter_progressive_val_score(
            dataset, model, metric, measure_time=True, measure_memory=True, step=step
        ):
            if verbose:
                print(checkpoint["Step"])
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


def plot_oml_iter_progressive(result):
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

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.close()

    return fig
