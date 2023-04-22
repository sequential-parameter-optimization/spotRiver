from river import time_series
from river import compose
from river import linear_model
from river import optim
from river import preprocessing
from river import metrics
from numpy.random import default_rng
import numpy as np
from numpy import array

from spotRiver.utils.features import get_weekday_distances
from spotRiver.utils.features import get_ordinal_date
from spotRiver.utils.features import get_month_distances
from spotRiver.utils.features import get_hour_distances
from spotRiver.evaluation.eval_oml import fun_eval_oml_iter_progressive
from spotRiver.evaluation.eval_oml import eval_oml_iter_progressive
from spotRiver.evaluation.eval_bml import eval_oml_horizon
from spotRiver.evaluation.eval_nowcast import eval_nowcast_model

from spotPython.hyperparameters.values import (
    assign_values,
)
from spotPython.hyperparameters.prepare import (
    get_dict_with_levels_and_types,
    get_one_config_from_var_dict,
    iterate_dict_values,
    convert_keys,
)

from spotPython.utils.transform import transform_hyper_parameter_values

import logging
import statistics
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
# configure the handler and formatter as needed
py_handler = logging.FileHandler(f"{__name__}.log", mode="w")
py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
# add formatter to the handler
py_handler.setFormatter(py_formatter)
# add handler to the logger
logger.addHandler(py_handler)


class HyperRiver:
    """
    Hyperparameter Tuning for River.

    Args:
        seed (int): seed.
            See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)

    """

    def __init__(self, seed=126, log_level=50):
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {
            "seed": None,
            "data": None,
            "step": 10_000,
            "horizon": None,
            "grace_period": None,
            "metric": metrics.MAE(),
            "metric_sklearn": mean_absolute_error,
            "weights": array([1, 0, 0]),
            "weight_coeff": 0.0,
            "log_level": log_level,
            "var_name": [],
            "var_type": [],
            "prep_model": None,
        }
        self.log_level = self.fun_control["log_level"]
        logger.setLevel(self.log_level)
        logger.info(f"Starting the logger at level {self.log_level} for module {__name__}:")

    # def get_month_distances(x):
    #     return {
    #         calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
    #         for month in range(1, 13)
    #     }

    # def get_ordinal_date(x):
    #     return {'ordinal_date': x['month'].toordinal()}
    def fun_nowcasting(self, X, fun_control=None):
        """Hyperparameter Tuning of the nowcasting model.

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
        lr = X[:, 0]
        intercept_lr = X[:, 1]
        hour = X[:, 2]
        weekday = X[:, 3]
        month = X[:, 4]

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
                ("features", extract_features),
                ("scale", preprocessing.StandardScaler()),
                (
                    "lin_reg",
                    linear_model.LinearRegression(
                        intercept_init=0, optimizer=optim.SGD(float(lr[i])), intercept_lr=float(intercept_lr[i])
                    ),
                ),
            )
            # eval:
            dates, metric, y_trues, y_preds = eval_nowcast_model(
                model, dataset=self.fun_control["data"], time_interval="Time"
            )
            z = metric.get()
            z_res = np.append(z_res, z)
        return z_res

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

                `weekday` (bool): If `True`, an weekday component is added.

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
                self.fun_control["data"],
                model,
                metric=self.fun_control["metric"],
                horizon=self.fun_control["horizon"],
                agg_func=statistics.mean,
            )
            z = res.get()
            z_res = np.append(z_res, z)
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
                agg_func=statistics.mean,
            )
            z = res.get()
            z_res = np.append(z_res, z)
        return z_res

    # def fun_HTR_iter_progressive(self, X, fun_control=None):
    #     """Hyperparameter Tuning of HTR model.
    #     See: https://riverml.xyz/0.15.0/api/tree/HoeffdingTreeRegressor/
    #     Parameters
    #     ----------
    #     grace_period
    #         Number of instances a leaf should observe between split attempts.
    #     max_depth
    #         The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    #     delta
    #         Significance level to calculate the Hoeffding bound. The significance level is given by
    #         `1 - delta`. Values closer to zero imply longer split decision delays.
    #     tau
    #         Threshold below which a split will be forced to break ties.
    #     leaf_prediction
    #         Prediction mechanism used at leafs. NOTE: order differs from the order in river!</br>
    #         - 'mean' - Target mean</br>
    #         - 'adaptive' - Chooses between 'mean' and 'model' dynamically</br>
    #         - 'model' - Uses the model defined in `leaf_model`</br>
    #     NOT IMPLEMENTED: leaf_model
    #         The regression model used to provide responses if `leaf_prediction='model'`. If not
    #         provided an instance of `river.linear_model.LinearRegression` with the default
    #         hyperparameters is used.
    #     model_selector_decay
    #         The exponential decaying factor applied to the learning models' squared errors, that
    #         are monitored if `leaf_prediction='adaptive'`. Must be between `0` and `1`. The closer
    #         to `1`, the more importance is going to be given to past observations. On the other hand,
    #         if its value approaches `0`, the recent observed errors are going to have more influence
    #         on the final decision.
    #     nominal_attributes
    #         List of Nominal attributes identifiers. If empty, then assume that all numeric attributes
    #         should be treated as continuous.
    #     splitter
    #         The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
    #         features and perform splits. Splitters are available in the `tree.splitter` module.
    #         Different splitters are available for classification and regression tasks. Classification
    #         and regression splitters can be distinguished by their property `is_target_class`.
    #         This is an advanced option. Special care must be taken when choosing different splitters.
    #         By default, `tree.splitter.TEBSTSplitter` is used if `splitter` is `None`.
    #     min_samples_split
    #         The minimum number of samples every branch resulting from a split candidate must have
    #         to be considered valid.
    #     binary_split
    #         If True, only allow binary splits.
    #     max_size
    #         The max size of the tree, in Megabytes (MB).
    #     memory_estimate_period
    #         Interval (number of processed instances) between memory consumption checks.
    #     stop_mem_management
    #         If True, stop growing as soon as memory limit is hit.
    #     remove_poor_attrs
    #         If True, disable poor attributes to reduce memory usage.
    #     merit_preprune
    #         If True, enable merit-based tree pre-pruning.

    #     fun_control
    #         Parameters that are are not tuned:
    #             1. `horizon`: (int)
    #             2. `grace_period`: (int) Initial period during which the metric is not updated.
    #                     This is to fairly evaluate models which need a warming up period to start
    #                     producing meaningful forecasts.
    #                     The value of this parameter is equal to the `horizon` by default.
    #             3. `data`: dataset. Default `AirlinePassengers`.

    #     Returns
    #     -------
    #     (float): objective function value. Mean of the MAEs of the predicted values.
    #     """
    #     self.fun_control.update(fun_control)
    #     try:
    #         X.shape[1]
    #     except ValueError:
    #         X = np.array([X])
    #     if X.shape[1] != 11:
    #         raise Exception
    #     grace_period = X[:, 0]
    #     max_depth = X[:, 1]
    #     delta = X[:, 2]
    #     tau = X[:, 3]
    #     leaf_prediction = X[:, 4]
    #     leaf_model = X[:, 5]
    #     model_selector_decay = X[:, 6]
    #     splitter = X[:, 7]
    #     min_samples_split = X[:, 8]
    #     binary_split = X[:, 9]
    #     max_size = X[:, 10]
    #     z_res = np.array([], dtype=float)
    #     dataset_list = self.fun_control["data"]
    #     for i in range(X.shape[0]):
    #         num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
    #         cat = compose.SelectType(str) | preprocessing.FeatureHasher(n_features=1000, seed=1)
    #         try:
    #             res = eval_oml_iter_progressive(
    #                 dataset=dataset_list,
    #                 step=self.fun_control["step"],
    #                 log_level=self.fun_control["log_level"],
    #                 metric=fun_control["metric"],
    #                 weight_coeff=fun_control["weight_coeff"],
    #                 models={
    #                     "HTR": (
    #                         (num + cat)
    #                         | tree.HoeffdingTreeRegressor(
    #                             grace_period=int(grace_period[i]),
    #                             max_depth=transform_power_10(int(max_depth[i])),
    #                             delta=float(delta[i]),
    #                             tau=float(tau[i]),
    #                             leaf_prediction=select_leaf_prediction(int(leaf_prediction[i])),
    #                             leaf_model=select_leaf_model(int(leaf_model[i])),
    #                             model_selector_decay=float(model_selector_decay[i]),
    #                             splitter=select_splitter(int(splitter[i])),
    #                             min_samples_split=int(min_samples_split[i]),
    #                             binary_split=int(binary_split[i]),
    #                             max_size=float(max_size[i]),
    #                         )
    #                     ),
    #                 },
    #             )
    #             logger.debug("res from eval_oml_iter_progressive: %s", res)
    #             y = fun_eval_oml_iter_progressive(res, metric=None, weights=self.fun_control["weights"])
    #         except Exception as err:
    #             y = np.nan
    #             print(f"Error in fun(). Call to evaluate failed. {err=}, {type(err)=}")
    #             print(f"Setting y to {y:.2f}.")
    #         z_res = np.append(z_res, y / self.fun_control["n_samples"])
    #     return z_res

    def fun_oml_iter_progressive(self, X, fun_control=None):
        """Hyperparameter Tuning of an arbitrary model.
        Returns
        -------
        (float): objective function value. Mean of the MAEs of the predicted values.
        """
        self.fun_control.update(fun_control)
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception
        var_dict = assign_values(X, self.fun_control["var_name"])
        z_res = np.array([], dtype=float)
        dataset_list = self.fun_control["data"]
        for values in iterate_dict_values(var_dict):
            values = convert_keys(values, self.fun_control["var_type"])
            print(values)
            values = get_dict_with_levels_and_types(fun_control=self.fun_control, v=values)
            values = transform_hyper_parameter_values(fun_control=self.fun_control, hyper_parameter_values=values)
            print(values)
            model = compose.Pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**values))
            try:
                res = eval_oml_iter_progressive(
                    dataset=dataset_list,
                    step=self.fun_control["step"],
                    log_level=self.fun_control["log_level"],
                    metric=fun_control["metric"],
                    weight_coeff=fun_control["weight_coeff"],
                    models={
                        self.fun_control["model_name"]: (model),
                    },
                )
                logger.debug("res from eval_oml_iter_progressive: %s", res)
                y = fun_eval_oml_iter_progressive(res, metric=None, weights=self.fun_control["weights"])
            except Exception as err:
                y = np.nan
                print(f"Error in fun(). Call to evaluate failed. {err=}, {type(err)=}")
                print(f"Setting y to {y:.2f}.")
            z_res = np.append(z_res, y / self.fun_control["n_samples"])
        return z_res

    def compute_y(self, df_eval):
        """Compute the objective function value.
        Args:
            df_eval (pd.DataFrame): DataFrame with the evaluation results.
        Returns:
            (float): objective function value. Mean of the MAEs of the predicted values.
        Examples:
            >>> df_eval = pd.DataFrame( [[1, 2, 3], [4, 5, 6]], columns=['Metric', 'CompTime (s)', 'Memory (MB)'])
            >>> weights = [1, 1, 1]
            >>> compute_y(df_eval, weights)
            4.0
        """
        # take the mean of the MAEs/ACCs of the predicted values and ignore the NaN values
        df_eval = df_eval.dropna()
        y_error = df_eval["Metric"].mean()
        logger.debug("y_error from eval_oml_horizon: %s", y_error)
        y_r_time = df_eval["CompTime (s)"].mean()
        logger.debug("y_r_time from eval_oml_horizon: %s", y_r_time)
        y_memory = df_eval["Memory (MB)"].mean()
        logger.debug("y_memory from eval_oml_horizon: %s", y_memory)
        weights = self.fun_control["weights"]
        logger.debug("weights from eval_oml_horizon: %s", weights)
        y = weights[0] * y_error + weights[1] * y_r_time + weights[2] * y_memory
        logger.debug("weighted res from eval_oml_horizon: %s", y)
        return y

    # def fun_oml_horizon_old(self, X, fun_control=None, return_model=False, return_df=False):
    #     """Hyperparameter Tuning of an arbitrary model.
    #     Returns
    #     -------
    #     (float): objective function value. Mean of the MAEs of the predicted values.
    #     """
    #     self.fun_control.update(fun_control)
    #     weights = self.fun_control["weights"]
    #     if len(weights) != 3:
    #         raise ValueError("The weights array must be of length 3.")
    #     try:
    #         X.shape[1]
    #     except ValueError:
    #         X = np.array([X])
    #     if X.shape[1] != len(self.fun_control["var_name"]):
    #         raise Exception
    #     var_dict = assign_values(X, self.fun_control["var_name"])
    #     z_res = np.array([], dtype=float)
    #     for values in iterate_dict_values(var_dict):
    #         values = convert_keys(values, self.fun_control["var_type"])
    #         values = get_dict_with_levels_and_types(fun_control=self.fun_control, v=values)
    #         values = transform_hyper_parameter_values(fun_control=self.fun_control, hyper_parameter_values=values)
    #         model = compose.Pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**values))
    #         if return_model:
    #             return model
    #         try:
    #             df_eval, df_preds = eval_oml_horizon(
    #                 model=model,
    #                 train=self.fun_control["train"],
    #                 test=self.fun_control["test"],
    #                 target_column=self.fun_control["target_column"],
    #                 horizon=self.fun_control["horizon"],
    #                 oml_grace_period=self.fun_control["oml_grace_period"],
    #                 metric=self.fun_control["metric_sklearn"],
    #             )
    #         except Exception as err:
    #             print(f"Error in fun_oml_horizon(). Call to eval_oml_horizon failed. {err=}, {type(err)=}")
    #         if return_df:
    #             return df_eval, df_preds
    #         try:
    #             y = self.compute_y(df_eval, weights)
    #         except Exception as err:
    #             y = np.nan
    #             print(f"Error in fun(). Call to evaluate failed. {err=}, {type(err)=}")
    #             print(f"Setting y to {y:.2f}.")
    #         z_res = np.append(z_res, y / self.fun_control["n_samples"])
    #     return z_res

    def check_weights(self):
        if len(self.fun_control["weights"]) != 3:
            raise ValueError("The weights array must be of length 3.")

    def check_X_shape(self, X):
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception

    def evaluate_model(self, model, fun_control):
        try:
            df_eval, df_preds = eval_oml_horizon(
                model=model,
                train=fun_control["train"],
                test=fun_control["test"],
                target_column=fun_control["target_column"],
                horizon=fun_control["horizon"],
                oml_grace_period=fun_control["oml_grace_period"],
                metric=fun_control["metric_sklearn"],
            )
        except Exception as err:
            print(f"Error in fun_oml_horizon(). Call to eval_oml_horizon failed. {err=}, {type(err)=}")
        return df_eval, df_preds

    def get_river_df_eval_preds(self, model):
        try:
            df_eval, df_preds = self.evaluate_model(model, self.fun_control)
        except Exception as err:
            print(f"Error in get_river_df_eval_preds(). Call to evaluate_model failed. {err=}, {type(err)=}")
            print("Setting df_eval and df.preds to np.nan")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def fun_oml_horizon_old(self, X, fun_control=None):
        z_res = np.array([], dtype=float)
        self.fun_control.update(fun_control)
        self.check_weights()
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in get_one_config_from_var_dict(var_dict, self.fun_control):
            if self.fun_control["prep_model"] is not None:
                model = compose.Pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**config))
            else:
                model = self.fun_control["core_model"](**config)
            try:
                df_eval, _ = self.evaluate_model(model, self.fun_control)
            except Exception as err:
                df_eval = np.nan
                print(f"Error in fun(). Call to evaluate failed. {err=}, {type(err)=}")
                print("Setting df_eval to np.nan.")
            try:
                y = self.compute_y(df_eval)
            except Exception as err:
                y = np.nan
                print(f"Error in fun(). Call to compute_y failed. {err=}, {type(err)=}")
                print("Setting y to np.nan.")
            z_res = np.append(z_res, y / self.fun_control["n_samples"])
        return z_res

    def fun_oml_horizon(self, X, fun_control=None):
        """
        This function calculates the horizon for a given set of data X and control parameters.

        :param X: numpy array of data
        :param fun_control: dictionary of control parameters
        :return: numpy array of horizon values
        """
        self.fun_control.update(fun_control)
        self.check_weights()
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        z_res = []
        for config in get_one_config_from_var_dict(var_dict, self.fun_control):
            if self.fun_control["prep_model"] is not None:
                model = compose.Pipeline(self.fun_control["prep_model"], self.fun_control["core_model"](**config))
            else:
                model = self.fun_control["core_model"](**config)
            try:
                df_eval, _ = self.evaluate_model(model, self.fun_control)
                y = self.compute_y(df_eval)
            except Exception as err:
                y = np.nan
                print(f"Error in fun(). Call to evaluate or compute_y failed. {err=}, {type(err)=}")
                print("Setting y to np.nan.")
            z_res.append(y / self.fun_control["n_samples"])
        return np.array(z_res)
