from sklearn.datasets import fetch_openml


def get_bike_sharing_data(train_size=0.6):
    """
    Fetches the Bike Sharing Demand dataset from OpenML and splits it into training and test sets.

    Args:
        train_size (float):
            The proportion of the dataset to include in the training set. Default value: 0.6
    Returns:
        (tuple): tuple containing:
            df (pd.DataFrame): The full dataset.
            train (pd.DataFrame): The training set.
            test (pd.DataFrame): The test set.

    Examples:
        >>> from spotRiver.data.bike_sharing import get_bike_sharing_data
        >>> df, train, test = get_bike_sharing_data(train_size=0.6)
    """

    bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas")
    df = bike_sharing.frame
    # Normalize the count column
    df["count"] = df["count"] / df["count"].max()
    # Replace heavy_rain with rain in the weather column
    df["weather"].replace(to_replace="heavy_rain", value="rain", inplace=True)
    n = df.shape[0]
    # Calculate the number of rows in the training set
    k = int(n * train_size)
    # Split the data into training and test sets
    train = df[0:k]
    test = df[k:n]
    return df, train, test
