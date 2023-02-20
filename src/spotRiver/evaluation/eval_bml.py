from numpy import round, floor, array, abs, max, mean, sum
from pandas import DataFrame
from pandas import concat
from pandas import Series
from datetime import datetime
import tracemalloc
from sklearn.pipeline import Pipeline
from sklearn import preprocessing as preprocessing_sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def evaluateModel(diff, memory, time):
    pos_sum = 0
    neg_sum = 0
    for e in diff:
        if e > 0:
            pos_sum += e
        else:
            neg_sum += e
    rmse = (diff**2).mean() ** 0.5
    mae = mean(abs(diff))
    res_dict = {"AIC": None, "BIC": None}
    res_dict = res_dict | {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "AbsDiff": round(sum(abs(array(diff))), 2),
        "Underestimation": round(pos_sum, 2),
        "Overestimation": round(abs(neg_sum), 2),
        "MaxResidual": round(max(abs(diff)), 2),
        "Memory (MB)": round(memory, 4),
        "CompTime (s)": round(time, 4),
    }
    return res_dict


def plot_results(df_eval=None, Metric=False, df_true=None, real_vs_predict=False):
    if df_eval is not None and Metric:
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16, 5), constrained_layout=True, sharex=True)
        ax1.plot(df_eval.index, df_eval["MAE"])
        ax1.set_title("Mean Absolute Error")
        ax2.plot(df_eval.index, df_eval["Memory (MB)"])
        ax2.set_title("Memory (MB)")
        ax3.plot(df_eval.index, df_eval["CompTime (s)"])
        ax3.set_title("Computation time (s)")
    if df_true is not None and real_vs_predict:
        plt.figure(figsize=(16, 5))
        plt.plot(df_true.index, df_true["Vibration"], label="Actual")
        plt.plot(df_true.index, df_true["Prediction"], label="Prediction")
        plt.title("Actual vs Prediction")
        plt.legend()
        plt.show()


def eval_bml(train=None, test=None, horizon=None, model=None):
    df_eval = DataFrame(
        columns=[
            "RMSE",
            "MAE",
            "AbsDiff",
            "Underestimation",
            "Overestimation",
            "MaxResidual",
            "Memory (MB)",
            "CompTime (s)",
        ]
    )
    series_preds = Series([])
    series_diffs = Series([])
    start = datetime.now()
    tracemalloc.start()
    if model is None:
        model = Pipeline([("scaler", preprocessing_sklearn.StandardScaler()), ("lr", LinearRegression())])
        model.fit(train.iloc[:, :-1], train.iloc[:, -1])
    current, peak = tracemalloc.get_traced_memory()
    end = datetime.now()
    time = (end - start).total_seconds()
    df_eval.loc[0] = Series(
        {
            "RMSE": None,
            "MAE": None,
            "AbsDiff": None,
            "Underestimation": None,
            "Overestimation": None,
            "MaxResidual": None,
            "Memory (MB)": round(peak / 10**6, 4),
            "CompTime (s)": round(time, 4),
        }
    )
    if horizon is None:
        for i in range(0, len(test)):
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict(array(test.iloc[i, :-1]).reshape(1, -1))
            preds = Series(forecast, index=[i])
            diffs = test.iloc[i, -1] - preds
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[i + 1] = Series(evaluateModel(diffs, (peak / 10**6), time))
            series_preds = concat([series_preds, preds])
            series_diffs = concat([series_diffs, diffs])
    if horizon is not None:
        if len(test) % horizon == 0:
            for i in range(0, (int(len(test) / horizon) - 1)):
                start = datetime.now()
                tracemalloc.start()
                forecast = model.predict(test.iloc[i * horizon : (i + 1) * horizon, :-1])
                preds = Series(forecast)
                diffs = test.iloc[i * horizon : (i + 1) * horizon, -1].values - preds
                current, peak = tracemalloc.get_traced_memory()
                end = datetime.now()
                time = (end - start).total_seconds()
                df_eval.loc[i + 1] = Series(evaluateModel(diffs, (peak / 10**6), time))
                series_preds = concat([series_preds, preds])
                series_diffs = concat([series_diffs, diffs])
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
        if len(test) % horizon != 0:
            length = floor(len(test) / horizon)
            for i in range(0, (int(length))):
                start = datetime.now()
                tracemalloc.start()
                forecast = model.predict(test.iloc[i * horizon : (i + 1) * horizon, :-1])
                preds = Series(forecast)
                diffs = test.iloc[i * horizon : (i + 1) * horizon, -1].values - preds
                current, peak = tracemalloc.get_traced_memory()
                end = datetime.now()
                time = (end - start).total_seconds()
                df_eval.loc[i + 1] = Series(evaluateModel(diffs, (peak / 10**6), time))
                series_preds = concat([series_preds, preds])
                series_diffs = concat([series_diffs, diffs])
            start = datetime.now()
            tracemalloc.start()
            forecast = model.predict(
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), :-1]
            )
            preds = Series(forecast)
            diffs = (
                test.iloc[int(length * horizon) : int((length * horizon + len(test) % horizon - 1)), -1].values - preds
            )
            current, peak = tracemalloc.get_traced_memory()
            end = datetime.now()
            time = (end - start).total_seconds()
            df_eval.loc[int(length)] = Series(evaluateModel(diffs, (peak / 10**6), time))
            series_preds = concat([series_preds, preds])
            series_diffs = concat([series_diffs, diffs])
            series_preds = series_preds.reset_index(drop=True)
            series_diffs = series_diffs.reset_index(drop=True)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs
