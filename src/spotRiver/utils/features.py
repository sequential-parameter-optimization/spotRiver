import calendar
import math


def get_month_distances(x):
    k = list(x.keys())[0]
    return {calendar.month_name[month]: math.exp(-((x[k].month - month) ** 2)) for month in range(1, 13)}


def get_weekday_distances(x):
    # Monday is the first day, i.e., 0:
    k = list(x.keys())[0]
    return {calendar.day_name[weekday]: math.exp(-((x[k].weekday() - weekday) ** 2)) for weekday in range(0, 7)}


def get_hour_distances(x):
    k = list(x.keys())[0]
    return {str(hour): math.exp(-((x[k].hour - hour) ** 2)) for hour in range(0, 24)}


def get_ordinal_date(x):
    k = list(x.keys())[0]
    return {"ordinal_date": x[k].toordinal()}
