from river import datasets
import pandas as pd
from tabulate import tabulate


def convert_to_df(dataset: datasets.base.Dataset, target_column: str, n_total=None) -> pd.DataFrame:
    """Converts a river dataset into a pandas DataFrame.

    Args:
        dataset (datasets.base.Dataset): The river dataset to be converted.
        target_column (str): The name of the target column in the resulting DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame representation of the given dataset.

    Example:
        >>> dataset = datasets.TrumpApproval()
            target_column = "Approval"
            df = convert_to_df(dataset, target_column)
            df.rename(columns={
                'date': 'ordinal_date',
                'Gallup': 'gallup',
                'Ipsos': 'ipsos',
                'Morning Consult': 'morning_consult',
                'Rasmussen': 'rasmussen',
                'YouGov': 'you_gov'},
                inplace=True)
            # Split the data into train and test sets
            train = df[:500]
            test = df[500:]
    """
    data_dict = {key: [] for key in list(dataset.take(1))[0][0].keys()}
    data_dict[target_column] = []
    if n_total is None:
        for x in dataset:
            for key, value in x[0].items():
                data_dict[key].append(value)
            data_dict[target_column].append(x[1])
    else:
        for x in dataset.take(n_total):
            for key, value in x[0].items():
                data_dict[key].append(value)
            data_dict[target_column].append(x[1])
    df = pd.DataFrame(data_dict)
    return df


def compare_two_tree_models(model1, model2, headers=["Parameter", "Default", "Spot"]):
    """Compares two tree models and returns a table of the differences.
    Args:
        model1 (Pipeline): A river model pipeline.
        model2 (Pipeline): A river model pipeline.
    Returns:
        str: A table of the differences between the two models.
    """
    keys = model1[1].summary.keys()
    values1 = model1[1].summary.values()
    values2 = model2[1].summary.values()
    tbl = []
    for key, value1, value2 in zip(keys, values1, values2):
        tbl.append([key, value1, value2])
    return tabulate(tbl, headers=headers, numalign="right", tablefmt="github")
