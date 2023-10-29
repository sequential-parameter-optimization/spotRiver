import pandas as pd
from spotRiver.fun.hyperriver import HyperRiver

def test_compute_y():
    # create two Hyperriver objects
    hr = HyperRiver()
    hr_2 = HyperRiver(weights=[1.0, 2.0, 3.0])

    # create a sample evaluation DataFrame
    df_eval = pd.DataFrame({
        "Metric": [0.1, 0.2, 0.3],
        "CompTime (s)": [1.0, 2.0, 3.0],
        "Memory (MB)": [10.0, 20.0, 30.0]
    })

    # compute the objective function value
    y = hr.compute_y(df_eval)
    y_2 = hr_2.compute_y(df_eval)

    # check that the result is correct: calculate the means of the columns and multiply them with the weights
    assert y == 1.0 * df_eval["Metric"].mean() + 0.0 * df_eval["CompTime (s)"].mean() + 0.0 * df_eval["Memory (MB)"].mean()
    assert y_2 == 1.0 * df_eval["Metric"].mean() + 2.0 * df_eval["CompTime (s)"].mean() + 3.0 * df_eval["Memory (MB)"].mean()