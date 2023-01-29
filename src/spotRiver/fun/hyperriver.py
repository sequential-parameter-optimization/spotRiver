from river import time_series
from river import compose
from river import linear_model
from river import optim
from river import preprocessing
from river import metrics
from numpy.random import default_rng
import numpy as np
from spotRiver.utils.features import get_weekday_distances
from spotRiver.utils.features import get_ordinal_date
from spotRiver.utils.features import get_month_distances
from spotRiver.utils.features import get_hour_distances


class HyperRiver:
    """
    Hyperparameter Tuning for River.

    Args:
        seed (int): seed.
            See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    """

    def __init__(self, seed=126):
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {"seed": None, "data": None, "horizon": None, "grace_period": None, "metric": metrics.MAE()}

    # def get_month_distances(x):
    #     return {
    #         calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
    #         for month in range(1, 13)
    #     }

    # def get_ordinal_date(x):
    #     return {'ordinal_date': x['month'].toordinal()}

    def fun_snarimax(self, X, fun_control=None):
        """Hyperparameter Tuning of the SNARIMAX model.
            SNARIMAX stands for (S)easonal (N)on-linear (A)uto(R)egressive (I)ntegrated (M)oving-(A)verage with
            e(X)ogenous inputs model.
        This model generalizes many established time series models in a single interface that can be
        trained online. It assumes that the provided training data is ordered in time and is uniformly spaced.
        It is made up of the following components:
        - S (Seasonal)
        - N (Non-linear): Any online regression model can be used, not necessarily a linear regression
            as is done in textbooks.
        - AR (Autoregressive): Lags of the target variable are used as features.
        - I (Integrated): The model can be fitted on a differenced version of a time series. In this
            context, integration is the reverse of differencing.
        - MA (Moving average): Lags of the errors are used as features.
        - X (Exogenous): Users can provide additional features. Care has to be taken to include
            features that will be available both at training and prediction time.

        Each of these components can be switched on and off by specifying the appropriate parameters.
        Classical time series models such as AR, MA, ARMA, and ARIMA can thus be seen as special
        parametrizations of the SNARIMAX model.

        This model is tailored for time series that are homoskedastic. In other words, it might not
        work well if the variance of the time series varies widely along time.

        Parameters of the hyperparameter vector:

            `p`: Order of the autoregressive part. This is the number of past target values that will be
                included as features.
            `d`: Differencing order.
            `q`: Order of the moving average part. This is the number of past error terms that will be included
                as features.
            `m`: Season length used for extracting seasonal features. If you believe your data has a seasonal
                pattern, then set this accordingly. For instance, if the data seems to exhibit a yearly seasonality,
                and that your data is spaced by month, then you should set this to `12`.
                Note that for this parameter to have any impact you should also set at least
                one of the `p`, `d`, and `q` parameters.
            `sp`:  Seasonal order of the autoregressive part. This is the number of past target values that will
                be included as features.
            `sd`: Seasonal differencing order.
            `sq`: Seasonal order of the moving average part. This is the number of past error terms that will be
                included as features.
            `lr` (float):
                learn rate of the linear regression model. A river `preprocessing.StandardScaler`
                piped with a river `linear_model.LinearRegression` will be used.
            `intercept_lr` (float): intercept of the the linear regression model. A river `preprocessing.StandardScaler`
                piped with a river `linear_model.LinearRegression` will be used.
            `hour` (bool): If `True`, an hourly component is added.
            `weekdy` (bool): If `True`, an weekday component is added.
            `month` (bool): If `True`, an monthly component is added.

        Args:
            X (array):
                Seven hyperparameters to be optimized. Here:

                `p` (int):
                    Order of the autoregressive part.
                    This is the number of past target values that will be included as features.

                `d` (int):
                    Differencing order.

                `q` (int):
                    Order of the moving average part.
                    This is the number of past error terms that will be included as features.

                `m` (int):
                    Season length used for extracting seasonal features.
                    If you believe your data has a seasonal pattern, then set this accordingly.
                    For instance, if the data seems to exhibit a yearly seasonality,
                    and that your data is spaced by month, then you should set this to `12`.
                    Note that for this parameter to have any impact you should also set
                    at least one of the `p`, `d`, and `q` parameters.

                `sp` (int):
                    Seasonal order of the autoregressive part.
                    This is the number of past target values that will be included as features.

                `sd` (int):
                    Seasonal differencing order.

                `sq`(int):
                    Seasonal order of the moving average part.
                    This is the number of past error terms that will be included as features.

                `lr` (float):
                    learn rate of the linear regression model. A river `preprocessing.StandardScaler`
                    piped with a river `linear_model.LinearRegression` will be used.

                `intercept_lr` (float): intercept of the the linear regression model.
                    A river `preprocessing.StandardScaler` piped with a river `linear_model.LinearRegression`
                    will be used.

                `hour` (bool): If `True`, an hourly component is added.

                `weekdy` (bool): If `True`, an weekday component is added.

                `month` (bool): If `True`, an monthly component is added.

            fun_control (dict):
                parameter that are not optimized, e.g., `horizon`. Commonly
                referred to as "design of experiments" parameters:

                1. `horizon`: (int)

                2. `grace_period`: (int) Initial period during which the metric is not updated.
                    This is to fairly evaluate models which need a warming up period to start
                    producing meaningful forecasts.
                    The value of this parameter is equal to the `horizon` by default.

                3. `data`: dataset. Default `AirlinePassengers`.

        Returns:
            (float): objective function value. Mean of the MAEs of the predicted values.
        """
        self.fun_control.update(fun_control)

        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != 12:
            raise Exception
        p = X[:, 0]
        d = X[:, 1]
        q = X[:, 2]
        m = X[:, 3]
        sp = X[:, 4]
        sd = X[:, 5]
        sq = X[:, 6]
        lr = X[:, 7]
        intercept_lr = X[:, 8]
        hour = X[:, 9]
        weekday = X[:, 10]
        month = X[:, 11]

        # TODO:
        # horizon = fun_control["horizon"]
        # future = [
        #   {"month": dt.date(year=1961, month=m, day=1)} for m in range(1, horizon + 1)
        # ]
        z_res = np.array([], dtype=float)
        for i in range(X.shape[0]):
            h_i = int(hour[i])
            w_i = int(weekday[i])
            m_i = int(month[i])
            # baseline:
            extract_features = compose.TransformerUnion(get_ordinal_date)
            if h_i:
                extract_features = compose.TransformerUnion(get_ordinal_date, get_hour_distances)
            if w_i:
                extract_features = compose.TransformerUnion(extract_features, get_weekday_distances)
            if m_i:
                extract_features = compose.TransformerUnion(extract_features, get_month_distances)
            model = compose.Pipeline(
                extract_features,
                time_series.SNARIMAX(
                    p=int(p[i]),
                    d=int(d[i]),
                    q=int(q[i]),
                    m=int(m[i]),
                    sp=int(sp[i]),
                    sd=int(sd[i]),
                    sq=int(sq[i]),
                    regressor=compose.Pipeline(
                        preprocessing.StandardScaler(),
                        linear_model.LinearRegression(
                            intercept_init=0,
                            optimizer=optim.SGD(float(lr[i])),
                            intercept_lr=float(intercept_lr[i]),
                        ),
                    ),
                ),
            )
            # eval:
            res = time_series.evaluate(
                self.fun_control["data"], model, metric=self.fun_control["metric"], horizon=self.fun_control["horizon"]
            )
            y = res.metrics
            z = 0.0
            for j in range(len(y)):
                z = z + y[j].get()
            z_res = np.append(z_res, z / len(y))
        return z_res

    def fun_hw(self, X, fun_control=None):
        """Hyperparameter Tuning of the HoltWinters model.

            Holt-Winters forecaster. This is a standard implementation of the Holt-Winters forecasting method.
            Certain parameterizations result in special cases, such as simple exponential smoothing.

        Args:
            X (array): five hyperparameters. The parameters of the hyperparameter vector are:

                1. `alpha`: Smoothing parameter for the level.
                2. `beta`: Smoothing parameter for the trend.
                3. `gamma`: Smoothing parameter for the seasonality.
                4. `seasonality`: The number of periods in a season.
                    For instance, this should be 4 for quarterly data, and 12 for yearly data.
                5. `multiplicative`: Whether or not to use a multiplicative formulation.

            fun_control (dict): Parameters that are are not tuned:

                1. `horizon`: (int)
                2. `grace_period`: (int) Initial period during which the metric is not updated.
                    This is to fairly evaluate models which need a warming up period to start
                    producing meaningful forecasts.
                    The value of this parameter is equal to the `horizon` by default.
                2. `data`: dataset. Default `AirlinePassengers`.

        Returns:
            (float): objective function value. Mean of the MAEs of the predicted values.
        """
        self.fun_control.update(fun_control)
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != 5:
            raise Exception

        alpha = X[:, 0]
        beta = X[:, 1]
        gamma = X[:, 2]
        seasonality = X[:, 3]
        multiplicative = X[:, 4]
        z_res = np.array([], dtype=float)
        for i in range(X.shape[0]):
            model = time_series.HoltWinters(
                alpha=alpha[i],
                beta=beta[i],
                gamma=gamma[i],
                seasonality=int(seasonality[i]),
                multiplicative=int(multiplicative[i]),
            )
            res = time_series.evaluate(
                self.fun_control["data"],
                model,
                metric=self.fun_control["metric"],
                horizon=self.fun_control["horizon"],
                grace_period=self.fun_control["grace_period"],
            )
            y = res.metrics
            z = 0.0
            for j in range(len(y)):
                z = z + y[j].get()
            z_res = np.append(z_res, z / len(y))
        return z_res
