from numpy.random import default_rng
import numpy as np
import pandas as pd
from numpy import array
from river import compose
from typing import Optional, Dict, Any, Tuple
from spotPython.hyperparameters.values import assign_values
from spotPython.hyperparameters.values import (
    generate_one_config_from_var_dict,
)
from spotPython.utils.eda import generate_config_id
from spotRiver.evaluation.eval_bml import eval_oml_horizon

import logging
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
        """Initialize the class.
        Args:
            seed (int): seed.
                See [Numpy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start)
            log_level (int): The level of logging to use. 0 = no logging, 50 = print only important
                            information. Defaults to 50.

        Returns:
            (NoneType): None
        """
        self.seed = seed
        self.rng = default_rng(seed=self.seed)
        self.fun_control = {
            "seed": None,
            "data": None,
            "step": 10_000,
            "horizon": None,
            "grace_period": None,
            "metric_river": None,
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

    def check_X_shape(self, X):
        """
        Check the shape of X.

        Args:
            X (np.ndarray): The input data.

        Returns:
            (NoneType): None

        Examples:
            >>> X = np.array([[1, 2, 3], [4, 5, 6]])
            >>> check_X_shape(X)
            >>> X = np.array([1, 2, 3])
            >>> check_X_shape(X)
            Traceback (most recent call last):
            ...
            Exception

        """
        try:
            X.shape[1]
        except ValueError:
            X = np.array([X])
        if X.shape[1] != len(self.fun_control["var_name"]):
            raise Exception

    def evaluate_model(self, model: object, fun_control: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluates a model using the eval_oml_horizon function.

        Args:
            model (object): The model to be evaluated.
            fun_control (dict): A dictionary containing the following keys:
                - train (pd.DataFrame): The training data.
                - test (pd.DataFrame): The testing data.
                - target_column (str): The name of the target column.
                - horizon (int): The horizon value.
                - oml_grace_period (int): The oml_grace_period value.
                - metric_sklearn (str): The metric to be used for evaluation.

        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing two dataframes:
                - df_eval: The evaluation dataframe.
                - df_preds: The predictions dataframe.

        Examples:
            >>> model = SomeModel()
            >>> fun_control = {
            ...     "train": train_data,
            ...     "test": test_data,
            ...     "target_column": "target",
            ...     "horizon": 5,
            ...     "oml_grace_period": 10,
            ...     "metric_sklearn": "accuracy"
            ... }
            >>> df_eval, df_preds = evaluate_model(model, fun_control)
        """
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
        """Get the evaluation and prediction dataframes for a river model.

        Args:
            model (object): The model to be evaluated.

        Returns:
            (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing two dataframes:
                - df_eval: The evaluation dataframe.
                - df_preds: The predictions dataframe.

        Examples:
            >>> model = SomeModel()
            >>> df_eval, df_preds = get_river_df_eval_preds(model)
        """
        try:
            df_eval, df_preds = self.evaluate_model(model, self.fun_control)
        except Exception as err:
            print(f"Error in get_river_df_eval_preds(). Call to evaluate_model failed. {err=}, {type(err)=}")
            print("Setting df_eval and df.preds to np.nan")
            df_eval = np.nan
            df_preds = np.nan
        return df_eval, df_preds

    def fun_oml_horizon(self, X: np.ndarray, fun_control: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        The objective function for hyperparameter tuning.

        This function takes in input data and a dictionary of control parameters to compute the objective function values for hyperparameter tuning.

        Args:
            X (np.ndarray): The input data.
            fun_control (dict, optional): A dictionary containing the following keys:
                - train (pd.DataFrame): The training data.
                - test (pd.DataFrame): The testing data.
                - target_column (str): The name of the target column.
                - horizon (int): The horizon value.
                - oml_grace_period (int): The oml_grace_period value.
                - metric_sklearn (str): The metric to be used for evaluation.

        Returns:
            (np.ndarray): The objective function values.

        Examples:
            >>> fun_oml_horizon(X,
                                fun_control={'train': train_data,
                                             'test': test_data,
                                              'target_column': 'y',
                                              'horizon': 5,
                                              'oml_grace_period': 10,
                                              'metric_sklearn': 'accuracy'})
            array([0.8, 0.85, 0.9])
        """
        logger.debug("X from eval_oml_horizon: %s", X)
        logger.debug("fun_control from eval_oml_horizon: %s", fun_control)
        z_res = []
        self.fun_control.update(fun_control)
        self.check_X_shape(X)
        var_dict = assign_values(X, self.fun_control["var_name"])
        for config in generate_one_config_from_var_dict(var_dict, self.fun_control):
            logger.debug("config from eval_oml_horizon: %s", config)
            config_id = generate_config_id(config)
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
