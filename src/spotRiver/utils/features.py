import calendar
import math
from datetime import datetime
from typing import Dict


def get_month_distances(x: Dict[str, datetime]) -> Dict[str, float]:
    """
    This function takes in a dictionary with a single key-value pair where the key is a string and the value is a datetime object.
    It returns a new dictionary where the keys are the names of the months and the values are the result of applying a Gaussian
    function to the difference between the month of the input datetime object and each month.

    Args:
        x (Dict[str, datetime]): A dictionary with a single key-value pair where the key is a string and the value is a
        datetime object.

    Returns:
        Dict[str, float]: A dictionary where the keys are the names of the months and the values are the result of
        applying a Gaussian function to the difference between the month of the input datetime object and each month.
    Example:
        >>> get_month_distances({"date": datetime(2020, 1, 1)})
        {'January': 0.6065306597126334, 'February': 0.36787944117144233,
        'March': 0.1353352832366127, 'April': 0.01831563888873418, 'May': 0.00033546262790251185,
        'June': 3.354626279025118e-06, 'July': 2.0611536224385582e-08,
        'August': 8.315287191035679e-11,
        'September': 2.0611536224385582e-13, 'October': 3.354626279025118e-16,
        'November': 3.354626279025118e-19, 'December': 2.0611536224385582e-22}
    """
    k = list(x.keys())[0]
    return {calendar.month_name[month]: math.exp(-((x[k].month - month) ** 2)) for month in range(1, 13)}


def get_weekday_distances(x):
    """This function takes in a dictionary with a single key-value pair where the key is a string and the value is
        a datetime object.
        It returns a new dictionary where the keys are the names of the days of the week and the values are the
        result of applying a Gaussian function to the difference between the weekday of the input datetime object
        and each day of the week.
    Args:
        x (Dict[str, datetime]): A dictionary with a single key-value pair where the key is a string and the
        value is a datetime object.
    Returns:
        Dict[str, float]: A dictionary where the keys are the names of the days of the week and the values
        are the result of applying a Gaussian function to the difference between the weekday of the input
        datetime object and each day of the week.
    Example:
        >>> get_weekday_distances({"date": datetime(2020, 1, 1)})
        {'Monday': 0.6065306597126334, 'Tuesday': 0.36787944117144233, 'Wednesday': 0.1353352832366127,
        'Thursday': 0.01831563888873418, 'Friday': 0.00033546262790251185, 'Saturday': 3.354626279025118e-06,
        'Sunday': 2.0611536224385582e-08}
    """
    # Monday is the first day, i.e., 0:
    k = list(x.keys())[0]
    return {calendar.day_name[weekday]: math.exp(-((x[k].weekday() - weekday) ** 2)) for weekday in range(0, 7)}


def get_hour_distances(x):
    """This function takes in a dictionary with a single key-value pair where the key is a string and the value
        is a datetime object.
        It returns a new dictionary where the keys are the names of the hours of the day and the values are
        the result of applying a Gaussian function to the difference between the hour of the input datetime
        object and each hour of the day.
    Args:
        x (Dict[str, datetime]): A dictionary with a single key-value pair where the key is a string and the
        value is a datetime object.
    Returns:
        Dict[str, float]: A dictionary where the keys are the names of the hours of the day and the values are
        the result of applying a Gaussian function to the difference between the hour of the input datetime object and each hour of the day.
    Example:
        >>> get_hour_distances({"date": datetime(2020, 1, 1)})
        {'0': 0.6065306597126334, '1': 0.36787944117144233, '2': 0.1353352832366127, '3': 0.01831563888873418,
        '4': 0.00033546262790251185, '5': 3.354626279025118e-06, '6': 2.0611536224385582e-08, '7': 8.315287191035679e-11,
        '8': 2.0611536224385582e-13, '9': 3.354626279025118e-16, '10': 3.354626279025118e-19,
        '11': 2.0611536224385582e-22, '12': 8.315287191035679e-26, '13': 2.0611536224385582e-29,
        '14': 3.354626279025118e-33, '15': 3.354626279025118e-37, '16': 2.0611536224385582e-41,
        '17': 8.315287191035679e-46, '18': 2.0611536224385582e-50, '19': 3.354626279025118e-55,
        '20': 3.354626279025118e-60, '21': 2.0611536224385582e-65, '22': 8.315287191035679e-71, '23': 2.0611536224385582e-76}
    """
    k = list(x.keys())[0]
    return {str(hour): math.exp(-((x[k].hour - hour) ** 2)) for hour in range(0, 24)}


def get_ordinal_date(x):
    """This function takes in a dictionary with a single key-value pair where the key is a string and the value is a datetime object.
        It returns a new dictionary where the keys are the string "ordinal_date" and the value is the ordinal date of the input datetime object.
    Args:
        x (Dict[str, datetime]): A dictionary with a single key-value pair where the key is a string and the value is a datetime object.
    Returns:
        Dict[str, float]: A dictionary where the keys are the string "ordinal_date" and the value is the ordinal date of the input datetime object.
    Example:
        >>> get_ordinal_date({"date": datetime(2020, 1, 1)})
        {'ordinal_date': 737424}
    """
    k = list(x.keys())[0]
    return {"ordinal_date": x[k].toordinal()}
