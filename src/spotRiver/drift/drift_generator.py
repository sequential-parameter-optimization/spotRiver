import numpy as np


def generate_drift(data, drift_values=[1.1, 10.0, 0.1, 1.1]):
    """
    Generates a drift array based on the number of rows in the input data and the specified drift values.

    Parameters:
        data (pandas.DataFrame or numpy.ndarray): The input data.
        drift_values (list of float): The drift values to use.

    Returns:
        numpy.ndarray: The generated drift array.
    """
    num_rows = data.shape[0]
    num_drift_values = len(drift_values)
    quotient, remain = divmod(num_rows, num_drift_values)

    quotient_array = [value for value in drift_values for _ in range(quotient)]
    remain_array = np.full(remain, drift_values[-1], dtype=float)

    drift = np.concatenate([quotient_array, remain_array])
    return drift
