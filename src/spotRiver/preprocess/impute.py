import pandas as pd
import numpy as np
from spotRiver.data.opm import fetch_opm
from sklearn.impute import SimpleImputer


def impute_opm(
    include_categorical: bool = False,
    data_home: str = "data",
    strategy: str = "most_frequent",
    columns: list[str] = ["lat", "lon"],
    archive_name: str = "opm_cat.csv",
    path_or_buf: str = "opm_cat.zip",
    write_csv: bool = True,
    return_df: bool = False,
) -> pd.DataFrame:
    """Impute missing values in OPM dataset.

    Args:
        include_categorical: Whether to include categorical features. Default is False.
        data_home: The directory to use as a data store. Default is "data".
        strategy: The imputation strategy to use. Can be one of "mean", "median", "most_frequent", or "constant". Default is "most_frequent".
        columns: A list of column names to impute. If None, impute all columns. Default is ["lat", "lon"].
        archive_name: The name of the archive file to write. Default is "opm_cat.csv".
        path_or_buf: The file path or buffer to write. Default is "opm_cat.zip".
        write_csv: Whether to write the imputed data to a CSV file. Default is True.
        return_df: Whether to return the imputed data as a DataFrame. Default is False.

    Returns:
        If `return_df` is True, returns a pandas DataFrame containing the imputed data.
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
