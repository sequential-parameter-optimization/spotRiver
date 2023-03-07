from river import stream

from . import base


class GenericData(base.GenericFileDataset):
    """Generic File Data Class

    Args:
        base (type): The parent class to inherit from

    Attributes:
        fraction (float): The fraction of data to use

    """

    def __init__(
        self, filename, target, n_features, n_samples, converters, parse_dates, directory, task=base.REG, fraction=1.0
    ):
        """Constructor for GenericData.

        Args:
            filename (str): The name of the file containing the data.
            target (str): The name of the target variable.
            n_features (int): The number of features in the data.
            n_samples (int): The number of samples in the data.
            converters (dict): A dictionary mapping column names to converter functions.
            parse_dates (list): A list of column names that should be parsed as dates.
            directory (str): The path to the directory containing the data.
            task (str, optional): The name of the task to be performed. Defaults to 'REG'.
            fraction (float, optional): The fraction of the data to use. Defaults to 1.0.
        """
        super().__init__(
            filename=filename,
            n_features=n_features,
            n_samples=n_samples,
            task=task,
            target=target,
            converters=converters,
            parse_dates=parse_dates,
            directory=directory,
        )
        self.fraction = fraction

    def __iter__(self):
        """Iterates over the data.

        Returns:
            Iterator: An iterator over the data.
        """
        return stream.iter_csv(
            self.path,
            target=self.target,
            converters=self.converters,
            parse_dates=self.parse_dates,
            fraction=self.fraction,
            seed=123,
        )
