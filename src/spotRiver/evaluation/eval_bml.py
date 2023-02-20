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


def evaluate_model(diff, memory, time):
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
    start_time = datetime.now()
    tracemalloc.start()
    if model is None:
        model = Pipeline([("scaler", preprocessing_sklearn.StandardScaler()), ("lr", LinearRegression())])
        model.fit(train.drop(columns="target"), train["target"])
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    comp_time = (datetime.now() - start_time).total_seconds()
    df_eval.loc[0] = Series(
        {
            "RMSE": None,
            "MAE": None,
            "AbsDiff": None,
            "Underestimation": None,
            "Overestimation": None,
            "MaxResidual": None,
            "Memory (MB)": round(peak_memory / 10**6, 4),
            "CompTime (s)": round(comp_time, 4),
        }
    )
    if horizon is None:
        for i in range(len(test)):
            series_preds, series_diffs = eval_one(
                df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False
            )
    if horizon is not None:
        for i in range(0, len(test), horizon):
            is_last = i + horizon >= len(test)
            series_preds, series_diffs = eval_one(df_eval, i, model, horizon, test, series_preds, series_diffs, is_last)
    df_true = test.copy()
    df_true["Prediction"] = series_preds
    df_true["Difference"] = series_diffs
    return df_eval, df_true, series_preds, series_diffs


def eval_one(df_eval, i, model, horizon, test, series_preds, series_diffs, is_last=False):
    start_time = datetime.now()
    tracemalloc.start()
    if is_last:
        test_slice = test.iloc[i:]
    else:
        test_slice = test.iloc[i : i + horizon]
    preds = Series(model.predict(test_slice.drop(columns="target")))
    diffs = test_slice["target"].reset_index(drop=True) - preds.reset_index(drop=True)
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    comp_time = (datetime.now() - start_time).total_seconds()
    df_eval.loc[i + 1] = Series(evaluate_model(diffs, (peak_memory / 10**6), comp_time))
    series_preds = concat([series_preds, preds])
    series_diffs = concat([series_diffs, diffs])
    return series_preds, series_diffs


def plot_results(df_eval=None, metric=False, df_true=None, real_vs_predict=False):
    if df_eval is not None and metric:
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
