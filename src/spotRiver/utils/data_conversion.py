from river import datasets
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from numpy.typing import ArrayLike
from math import inf


def convert_to_df(dataset: datasets.base.Dataset, target_column: str = "y", n_total: int = None) -> pd.DataFrame:
    """Converts a river dataset into a pandas DataFrame.

    Args:
        dataset (datasets.base.Dataset):
            The river dataset to be converted.
        target_column (str):
            The name of the target column in the resulting DataFrame.
            Defaults to "y".
        n_total (int, optional):
            The number of samples to be converted.
            If set to None or inf, the full dataset is converted.
            Defaults to None, i.e, the full dataset is converted.

    Returns:
        (pd.DataFrame): A pandas DataFrame representation of the given dataset.

    Examples:
        >>> from river import datasets
            from spotRiver.utils.data_conversion import convert_to_df
            dataset = datasets.TrumpApproval()
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
    if n_total is None or n_total == inf:
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
        (str): A table of the differences between the two models.
    """
    keys = model1[1].summary.keys()
    values1 = model1[1].summary.values()
    values2 = model2[1].summary.values()
    tbl = []
    for key, value1, value2 in zip(keys, values1, values2):
        tbl.append([key, value1, value2])
    return tabulate(tbl, headers=headers, numalign="right", tablefmt="github")


def rename_df_to_xy(df, target_column="y"):
    """Renames the columns of a DataFrame to x1, x2, ..., xn, y.

    Args:
        df (pd.DataFrame):
            The DataFrame to be renamed.
        target_column (str, optional):
            The name of the target column. Defaults to "y".

    Returns:
        (pd.DataFrame): The renamed DataFrame.

    Examples:
        >>> from spotRiver.utils.data_conversion import rename_df_to_xy
            df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]
        })
        >>> df = rename_df_to_xy(df, "target")
        >>> print(df)
           x1  x2  y
        0   1   4  7
        1   2   5  8
        2   3   6  9
    """
    n_features = len(df.columns) - 1
    df.columns = [f"x{i}" for i in range(1, n_features + 1)] + [target_column]
    return df


def split_df(
    dataset: pd.DataFrame, test_size: float, seed: int, stratify: ArrayLike, shuffle=True, target_type: str = None
) -> tuple:
    """
    Split a pandas DataFrame into a training and a test set.

    Args:
        dataset (pd.DataFrame):
            The input data set.
        test_size (float):
            The percentage of the data set to be used as test set.
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
            If train_size is also None, it will be set to 0.25.
        target_type (str):
            The type of the target column. Can be "int", "float" or None.
            If None, the type of the target column is not changed.
            Otherwise, the target column is converted to the specified type.
        seed (int):
            The seed for the random number generator.
        stratify (ArrayLike):
            The array of target values.
        shuffle (bool, optional):
            Whether or not to shuffle the data before splitting. Defaults to True.

    Returns:
        tuple: The tuple (train, test, n_samples).

    Examples:
        >>> from spotRiver.utils.data_conversion import split_df
            df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [7, 8, 9]})
            train, test, n_samples = split_df(df, 0.2, "int", 42)

    """
    # Rename the columns of a DataFrame to x1, x2, ..., xn, y.
    # From now on we assume that the target column is called "y":
    df = rename_df_to_xy(df=dataset, target_column="y")
    if target_type == "float":
        df["y"] = df["y"].astype(float)
    elif target_type == "int":
        df["y"] = df["y"].astype(int)
    else:
        pass
    target_column = "y"
    # split the data set into a training and a test set,
    # where the test set is a percentage of the data set given as test_size:
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    # Split the data into training and test sets
    # test_size is the percentage of the data that should be held over for testing
    # random_state is a seed for the random number generator to make your train and test splits reproducible
    train_features, test_features, train_target, test_target = train_test_split(
        X, Y, test_size=test_size, random_state=seed, shuffle=shuffle, stratify=stratify
    )
    # combine the training features and the training target into a training DataFrame
    train = pd.concat([train_features, train_target], axis=1)
    test = pd.concat([test_features, test_target], axis=1)
    n_samples = train.shape[0] + test.shape[0]
    return train, test, n_samples
