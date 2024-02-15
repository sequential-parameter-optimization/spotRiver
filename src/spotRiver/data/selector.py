from river import datasets


def data_selector(
    data_set,
) -> tuple:
    """
    Selects the river data set to be used.

    Args:
        data_set (str):
            Name of the data set to be used. Can be one of the following:
            - "Bananas"
            - "CreditCard"
            - "Elec2"
            - "Higgs"
            - "HTTP"
            - "Phishing"

    Returns:
        dataset (object):
            Data set to use. This is a dataset object from the river library.
        n_samples (int):
            Number of samples in the data set.

    Examples:
        >>> from spotPython.data.selector import data_selector
            dataset, n_samples = data_selector("Phishing")

    Notes:
        - The Higgs data set is very large and may take a long time to load.

    """
    dataset = None
    if data_set == "Bananas":
        dataset = datasets.Bananas()
        n_samples = 5300
    elif data_set == "CreditCard":
        dataset = datasets.CreditCard()
        n_samples = 284_807
    elif data_set == "Elec2":
        dataset = datasets.Elec2()
        n_samples = 45_312
    elif data_set == "Higgs":
        dataset = datasets.Higgs()
        n_samples = 11_000_000
    elif data_set == "HTTP":
        dataset = datasets.HTTP()
        n_samples = 567_498
    elif data_set == "Phishing":
        dataset = datasets.Phishing()
        n_samples = 1250
    else:
        raise ValueError(f"Data set '{data_set}' not found.")
    return dataset, n_samples


def get_train_test_from_data_set(df, n_total, test_size, target_column="y") -> tuple:
    """Converts a data set to a data frame with target column
        and splits it into training and test sets.

    Args:
        df (DataFrame):
            data set to be used.
        n_total (int):
            total number of samples to be used in the data set.
        test_size (float):
            percentage of the data set to be used as test set.
        target_column (str, optional):
            name of the target column. Defaults to "y".

    Returns:
        train:
            training data set.
        test:
            test data set.
        n_samples:
            total number of samples (train and test) in the data set.

    """
    n_features = len(df.columns) - 1
    df.columns = [f"x{i}" for i in range(1, n_features + 1)] + ["y"]
    df["y"] = df["y"].astype(int)
    # update n_samples to the actual number of samples in the data set,
    # because n_total might be smaller than n_samples which results in a smaller data set:
    test_size = float(test_size)
    n_samples = len(df)
    n_train = int((1.0 - test_size) * n_samples)
    train = df[:n_train]
    test = df[n_train:]
    return train, test, n_samples
