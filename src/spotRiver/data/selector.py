from river import datasets
from spotRiver.data.csvdataset import CSVDataset
from spotRiver.utils.data_conversion import convert_to_df


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
            - "AirlinePassengers"
            - "Bikes"
            - "ChickWeights"
            - "Taxis"
            - "TrumpApproval"
            - "WaterFlow"
            - "WebTraffic"

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
    # TODO: Check and update the number of samples for each data set.
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
    elif data_set == "AirlinePassengers":
        dataset = datasets.AirlinePassengers()
        n_samples = 144
    elif data_set == "Bikes":
        dataset = datasets.Bikes()
        n_samples = 182470
    elif data_set == "ChickWeights":
        dataset = datasets.ChickWeights()
        n_samples = 578
    elif data_set == "Taxis":
        dataset = datasets.Taxis()
        n_samples = 1458644
    elif data_set == "TrumpApproval":
        dataset = datasets.TrumpApproval()
        n_samples = 1_000
    elif data_set == "WaterFlow":
        dataset = datasets.WaterFlow()
        n_samples = 1_000
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


def get_river_dataset_from_name(
    data_set_name,
    n_total=None,
    river_datasets=None,
):
    """Converts a data set name to a pandas DataFrame.

    Args:
        data_set_name (str):
            The name of the data set.
            If the data set name is not in river_datasets, the data set is assumed to be a CSV file.
        n_total (int):
            The number of samples to be used from the data set.
            If n_total is None, the full data set is used.
            Defaults to None.
        river_datasets (list):
            A list of the available river data sets.
            If the data set name is not in river_datasets,
            the data set is assumed to be a CSV file.

    Returns:
        pd.DataFrame:
            The data set as a pandas DataFrame.
        n_samples (int):
            The number of samples in the data set.
    """
    print(f"data_set_name: {data_set_name}")
    print("river_datasets: ", river_datasets)
    # data_set ends with ".csv" or data_set ends with ".pkl":
    if data_set_name.endswith(".csv"):
        print(f"data_set_name: {data_set_name}")
        dataset = CSVDataset(filename=data_set_name, directory="./userData/").data
        n_samples = dataset.shape[0]
    elif data_set_name in river_datasets:
        dataset, n_samples = data_selector(
            data_set=data_set_name,
        )
        # convert the river datasets to a pandas DataFrame, the target column
        # of the resulting DataFrame is target_column
        dataset = convert_to_df(dataset, target_column="y", n_total=n_total)
    return dataset, n_samples
