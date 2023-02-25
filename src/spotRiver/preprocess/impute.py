import pandas as pd
import numpy as np
from spotRiver.data.opm import fetch_opm
from sklearn.impute import SimpleImputer


def impute_opm(
    include_categorical=False,
    data_home="data",
    strategy="most_frequent",
    columns=["lat", "lon"],
    archive_name="opm_cat.csv",
    path_or_buf="opm_cat.zip",
    write_csv=True,
    return_df=False,
):
    """Imputes missing values in a dataframe using a given strategy.

    Parameters:
        include_categorical (bool): Whether to include categorical features in the dataframe.
        data_home (str): The path to the data directory.
        strategy (str): The imputation strategy to use. One of "mean", "median", "most_frequent" or "constant".
        columns (list of str): The names of the columns to impute. If None, all columns are imputed.
        archive_name (str): The name of the csv file to write the imputed dataframe to.
        path_or_buf (str): The path or buffer to write the compressed csv file to.
        write_csv (bool): Whether to write the imputed dataframe to a csv file.
        return_df (bool): Whether to return the imputed dataframe as output.

    Returns:
        pd.DataFrame: The imputed dataframe if return_df is True. Otherwise None.

    Raises:
        ValueError: If strategy is not one of the valid options or if columns are not in the dataframe.
    """
    # Validate input parameters
    valid_strategies = ["mean", "median", "most_frequent", "constant"]
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}.")
    # Fetch and concatenate data
    X, y = fetch_opm(include_categorical=include_categorical, data_home=data_home, return_X_y=True)
    df = pd.concat([X, y], axis=1)
    # Impute missing values
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
    if columns is None:
        # Impute all columns
        df[:] = imp.fit_transform(df)
    else:
        # Impute only specified columns
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Invalid column: {col}. Not in dataframe.")
            df[col] = imp.fit_transform(np.array(df[col]).reshape(-1, 1))
    # Write csv file if requested
    if write_csv:
        compression_opts = dict(method="zip", archive_name=archive_name)
        df.to_csv(path_or_buf, index=False, compression=compression_opts)
    # Return dataframe if requested
    if return_df:
        return df
