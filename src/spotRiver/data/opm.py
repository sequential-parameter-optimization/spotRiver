"""Office of Policy and Management dataset

The original database is available from CT's OPM

    https://portal.ct.gov/OPM/IGPP/Publications/Real-Estate-Sales-Listing

The data contains 985,862 observations of up to 14 variables.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.utils import Bunch
from typing import Union, Tuple
from urllib.request import urlretrieve

from spotRiver.data.base import get_data_home

logger = logging.Logger(__name__)

OPM_URL = "https://data.ct.gov/api/views/5mzw-sjtu/rows.csv?accessType=DOWNLOAD"
OPM_DTYPE = {
    "Address": np.dtype("O"),
    "Assessed Value": np.dtype("float64"),
    "Assessor Remarks": np.dtype("O"),
    "List Year": np.dtype("int64"),
    "Location": np.dtype("O"),
    "Non Use Code": np.dtype("O"),
    "OPM remarks": np.dtype("O"),
    "Property Type": np.dtype("O"),
    "Residential Type": np.dtype("O"),
    "Sale Amount": np.dtype("float64"),
    "Sales Ratio": np.dtype("float64"),
    "Serial Number": np.dtype("int64"),
    "Town": np.dtype("O"),
}


def fetch_opm(
    *,
    data_home: Union[str, Path] = None,
    download_if_missing: bool = True,
    return_X_y: bool = False,
    include_numeric: bool = True,
    include_categorical: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame, Bunch]:
    """Fetch the OPM dataset from the Connecticut Open Data portal.

    Parameters
    ----------
    data_home : str or pathlib.Path, default=None
        Specify another download and cache folder for the datasets. By default
        all spotRiver data is stored in '~/spotriver_data' subfolders.
    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        rather than trying to download the data from the source site.
    return_X_y : bool, default=False
        If True, return `(X, y)` instead of a `Bunch` object. See
        `sklearn.utils.Bunch` for more information.
    include_numeric : bool, default=True
        If True, include numeric columns in the output. Numeric columns include
        'List Year', 'Assessed Value', 'Sale Amount', 'Sales Ratio', 'lat', 'lon',
        and 'timestamp_rec'.
    include_categorical : bool, default=False
        If True, include categorical columns in the output. Categorical columns
        include 'Town', 'Address', 'Property Type', 'Residential Type',
        'Non Use Code', 'Assessor Remarks', and 'OPM remarks'. Columns with fewer
        than 200 unique values will be treated as categorical.

    Returns
    -------
    Bunch or Tuple[pd.DataFrame, pd.Series] or pd.DataFrame
        If `return_X_y` is False, return a `Bunch` object with the following
        attributes:
            * data : pd.DataFrame of shape (n_samples, n_features)
                The feature matrix.
            * target : pd.Series of shape (n_samples,)
                The target vector.
            * DESCR : str
                A short description of the dataset.
        If `return_X_y` is True, return a tuple `(X, y)` where `X` is the feature
        matrix and `y` is the target vector. If only numeric or categorical
        columns are included in the output, return a pd.DataFrame instead of a
        Bunch.

    Examples
    --------
    >>> from spotRiver.data import fetch_opm
        # Fetch the OPM dataset and return a pandas DataFrame
        opm_df = fetch_opm()
        # Fetch the OPM dataset, include categorical columns, and return a Bunch object
        opm_data = fetch_opm(include_numeric=False, include_categorical=True, return_X_y=False)
        # Fetch the OPM dataset, include numeric and categorical columns, and return a tuple of pandas DataFrames
        X, y = fetch_opm(include_categorical=True, return_X_y=True)
    """
    filename = get_data_home(data_home=data_home) / "opm_2001-2020.csv"
    if not filename.is_file():
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")
        logger.info(f"Downloading OPM dataset to '{filename}'.")
        urlretrieve(url=OPM_URL, filename=filename)
    # FIXME: Add hash check for download.

    df = pd.read_csv(filename, dtype=OPM_DTYPE, parse_dates=["Date Recorded"])

    # Collect rows (observations) we want to keep and subset only once.
    #
    # This might look kind of ugly but is much more efficient than making copy
    # after copy of the (largish) data frame `df`.
    idx = (
        (df["Date Recorded"] >= "2001-09-30")
        & (df["Assessed Value"] >= 2000)
        & (df["Assessed Value"] <= 1e8)
        & (df["Sale Amount"] >= 2000)
        & (df["Sale Amount"] <= 2e8)
    )
    logger.debug(f"Removing {len(idx) - idx.sum()} rows for constraint violations.")

    # Now keep only those rows which we selected with `idx`, sort the values by
    # the date on which they were recorded and then reset the index.
    df = df.loc[idx].sort_values(by="Date Recorded").reset_index(drop=True)

    cols = []
    if include_numeric:
        # Extract latitude and longitude from Location field.
        # Converting to float32 looses precision.
        df[["lon", "lat"]] = df["Location"].str.extract(r"POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)").astype("float")

        # Check if points are inside the bounding box for CT.
        # Bounding box taken from https://anthonylouisdagostino.com/bounding-boxes-for-all-us-states/
        outside_bbox = (
            (df["lon"] < -73.727775) | (df["lon"] > -71.786994) | (df["lat"] < 40.980144) | (df["lat"] > 42.050587)
        )
        df.loc[outside_bbox, ["lon", "lat"]] = np.nan
        logger.debug(f"Found {outside_bbox.sum()} locations outside of CT's bounding box.")

        # Convert types to smaller types to save some space.
        df["List Year"] = df["List Year"].astype("int16")

        # Add timestamp column by converting the Date Recorded to nanoseconds since epoch
        # and divide by 1e9 to go from nanoseconds to seconds since epoch.
        df["timestamp_rec"] = df["Date Recorded"].astype("int64") // 1e9

        # Converting `Assessed Value` to float32 changes 159 values
        df["Assessed Value"] = df["Assessed Value"].astype("int32")

        # Converting `Assessed Value` to int32/float32 changes 175/222 values
        # df["Sale Amount"] = df["Sale Amount"].astype("int32")

        cols.extend(["List Year", "Assessed Value", "Sale Amount", "Sales Ratio", "lat", "lon", "timestamp_rec"])

    # FIXME: We probably want to invest some time into deriving more meaninfgul
    # categorical variables from some of these. Especially the remarks columns
    # would benefit from a BoW approach and the Address column really carries very
    # little information.
    if include_categorical:
        categorical_columns = [
            "Town",
            "Address",
            "Property Type",
            "Residential Type",
            "Non Use Code",
            "Assessor Remarks",
            "OPM remarks",
        ]
        cols.extend(categorical_columns)

        for cat_col in categorical_columns:
            df[cat_col] = df[cat_col].fillna("Unknown")
            # If there less than 200 unique values, convert to "category"
            # instead of storing as a string to save space.
            if df[cat_col].nunique() < 200:
                df[cat_col] = df[cat_col].astype("category")

    if len(cols) == 0:
        raise Exception("No columns selected. Did you set both `include_numeric` and `include_categorical` to False?")

    X = df[cols]
    # y = df["Sale Amount"]
    y = X.pop("Sale Amount")

    if return_X_y:
        return (X, y)
    return Bunch(data=X, target=y)


__all__ = ["fetch_opm"]
