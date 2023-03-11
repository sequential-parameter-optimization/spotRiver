import pandas as pd
import numpy as np


def corrplot(df: pd.DataFrame, numeric_only=True) -> None:
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.
        The function takes a DataFrame df as input and generates a graphical correlation matrix
        for each pair of columns in the dataframe.
        The upper triangle of the correlation matrix is masked out and set to NaN values.
        The resulting matrix is then styled and returned as a heatmap with colors
        ranging from blue (for negative correlations) to red (for positive correlations).

    Input:
        df: pandas DataFrame
        numeric_only: bool, default True.
            Include only float, int or boolean data.

    Example:
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        corrplot(X)

    """

    # Compute the correlation matrix
    corr = df.corr(numeric_only=numeric_only)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set values in the upper triangle to NaN
    corr[mask] = np.nan

    # Apply styling to the correlation matrix and return it as a heatmap
    return corr.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1).highlight_null(color="#f1f1f1")
