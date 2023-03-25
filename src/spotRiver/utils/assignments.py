import numpy as np


def assign_values(X: np.array, var_list: list):
    """
    This function takes an np.array X and a list of variable names as input arguments
    and returns a dictionary where the keys are the variable names and the values are assigned from X.

    Parameters:
        X (np.array): A 2D numpy array where each column represents a variable.
        var_list (list): A list of strings representing variable names.

    Returns:
        dict: A dictionary where keys are variable names and values are assigned from X.

    Example:
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> var_list = ['a', 'b']
        >>> result = assign_values(X, var_list)
        >>> print(result)
        {'a': array([1, 3, 5]), 'b': array([2, 4, 6])}
    """
    result = {}
    for i, var_name in enumerate(var_list):
        result[var_name] = X[:, i]
    return result


def iterate_dict_values(var_dict: dict):
    """
    This function takes a dictionary of variables as input arguments and returns an iterator that yields the values from the arrays in the dictionary.

    Parameters:
        var_dict (dict): A dictionary where keys are variable names and values are numpy arrays.

    Returns:
        iterator: An iterator that yields the values from the arrays in the dictionary.

    Example:
        >>> import numpy as np
        >>> var_dict = {'a': np.array([1, 3, 5]), 'b': np.array([2, 4, 6])}
        >>> for values in iterate_dict_values(var_dict):
        ...     print(values)
        {'a': 1, 'b': 2}
        {'a': 3, 'b': 4}
        {'a': 5, 'b': 6}
    """
    n = len(next(iter(var_dict.values())))
    for i in range(n):
        yield {key: value[i] for key, value in var_dict.items()}


def convert_keys(d: dict, var_type: list):
    """
    Convert values in a dictionary to integers based on a list of variable types.

    This function takes a dictionary `d` and a list of variable types `var_type` as arguments. For each key in the dictionary,
    if the corresponding entry in `var_type` is not equal to `"num"`, the value associated with that key is converted to an integer.

    Args:
        d (dict): The input dictionary.
        var_type (list): A list of variable types. If the entry is not `"num"` the corresponding
        value will be converted to the type `"int"`.

    Returns:
        dict: The modified dictionary with values converted to integers based on `var_type`.

    Example:
        >>> d = {'a': '1.1', 'b': '2', 'c': '3.1'}
        >>> var_type = ["int", "num", "int"]
        >>> convert_keys(d, var_type)
        {'a': 1, 'b': '2', 'c': 3}
    """
    keys = list(d.keys())
    for i in range(len(keys)):
        if var_type[i] not in ["num", "float"]:
            d[keys[i]] = int(d[keys[i]])
    return d
