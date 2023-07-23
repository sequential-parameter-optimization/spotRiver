import numpy as np
import pandas as pd


def generate_drift(data: pd.DataFrame, drift_values=[1.1, 10.0, 0.1, 1.1]) -> np.ndarray:
    """
    Generates a drift array based on the number of rows in the input data and the specified drift values.

    Args:
        data (pd.DataFrame or np.ndarray): The input data.
        drift_values (list of float): The drift values to use.

    Returns:
        (np.ndarray): The generated drift array.

    Examples:
        >>> import numpy as np
        >>> from spotRiver.drift.drift_generator import generate_drift
        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> generate_drift(data, drift_values=[1.1, 10.0, 0.1, 1.1])
        array([ 1.1, 10. ,  0.1,  1.1])

    """
    num_rows = data.shape[0]
    num_drift_values = len(drift_values)
    quotient, remain = divmod(num_rows, num_drift_values)

    quotient_array = [value for value in drift_values for _ in range(quotient)]
    remain_array = np.full(remain, drift_values[-1], dtype=float)

    drift = np.concatenate([quotient_array, remain_array])
    return drift
