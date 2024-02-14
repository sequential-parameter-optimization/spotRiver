from river import datasets
from spotRiver.data.generic import GenericData
from spotRiver.utils.data_conversion import convert_to_df


def data_selector(
    data_set,
    filename="PhishingData.csv",
    directory="./userData",
    target="is_phishing",
    n_samples=1_250,
    n_features=9,
    converters={
        "empty_server_form_handler": float,
        "popup_window": float,
        "https": float,
        "request_from_other_domain": float,
        "anchor_from_other_domain": float,
        "is_popular": float,
        "long_url": float,
        "age_of_domain": int,
        "ip_in_url": int,
        "is_phishing": lambda x: x == "1",
    },
    parse_dates={"Time": "%Y-%m-%d %H:%M:%S%z"},
):
    """
    Selects the data set to be used.

    Args:
        data_set (str, optional):
            Data set to use. Defaults to "Phishing".
        filename (str, optional):
            Name of the file to read. Defaults to "user_data.csv".
        directory (str, optional):
            Directory where the file is located. Defaults to "./userData".
        target (str, optional):
            Name of the target column. Defaults to "Consumption".
        n_features (int, optional):
            Number of features. Defaults to 1.
        converters (dict, optional):
            Dictionary of functions for converting values in certain columns. Defaults to {"Consumption": float}.
        parse_dates (dict, optional):
            Dictionary of functions for parsing values in certain columns. Defaults to {"Time": "%Y-%m-%d %H:%M:%S%z"}.


    Returns:
        dataset (object):
            Data set to use.
        n_samples (int):
            Number of samples in the data set.
    Raises:
        ValueError:
            If data_set is not "Bananas" or "CreditCard" or "Elec2" or "Higgs" or
            "HTTP" or "MaliciousURL" or "Phishing" or "SMSSpam" or "SMTP" or "TREC07".

    Examples:
        >>> dataset, n_samples = data_selector("Phishing")


    """
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
    elif data_set == "MaliciousURL":
        dataset = datasets.MaliciousURL()
        n_samples = 2_396_130
    elif data_set == "Phishing":
        dataset = datasets.Phishing()
        n_samples = 1250
    elif data_set == "SMSSpam":
        dataset = datasets.SMSSpam()
        n_samples = 5574
    elif data_set == "SMTP":
        dataset = datasets.SMTP()
        n_samples = 95_156
    elif data_set == "TREC07":
        n_samples = 75_419
        dataset = datasets.TREC07()
    else:
        dataset = GenericData(
            filename=filename,
            directory=directory,
            target=target,
            n_features=n_features,
            n_samples=n_samples,
            converters=converters,
            parse_dates=parse_dates,
        )
        n_samples = dataset.n_samples
    return dataset, n_samples


def get_train_test_from_data_set(dataset, n_total, test_size, target_column="y"):
    """Converts a data set to a data frame with target column
        and splits it into training and test sets.

    Args:
        dataset:
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
    """
    df = convert_to_df(dataset, target_column=target_column, n_total=n_total)
    df.columns = [f"x{i}" for i in range(1, dataset.n_features + 1)] + ["y"]
    df["y"] = df["y"].astype(int)
    # update n_samples to the actual number of samples in the data set,
    # because n_total might be smaller than n_samples which results in a smaller data set:
    test_size = float(test_size)
    n_samples = len(df)
    n_train = int((1.0 - test_size) * n_samples)
    train = df[:n_train]
    test = df[n_train:]
    return train, test
