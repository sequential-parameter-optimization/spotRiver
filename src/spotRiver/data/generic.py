from river import stream
from . import base
from typing import Dict, List, Union

class GenericData(base.GenericFileDataset):
    """A class for handling generic data.

    This class inherits from the base.GenericFileDataset class and provides an interface for handling generic data.

    Args:
        filename (str): The name of the file containing the data.
        target (str): The name of the target column.
        n_features (int): The number of features in the dataset.
        n_samples (int): The number of samples in the dataset.
        converters (Dict[str, callable]): A dictionary of functions for converting column data.
        parse_dates (List[str]): A list of column names to parse as dates.
        directory (str): The directory where the file is located.
        task (str): The type of task. Default is base.REG for regression.
        fraction (float): The fraction of the data to use. Default is 1.0 for all data.

    Returns:
        (Generator): An iterator over the data in the file.

    Examples:
        >>> from spotRiver.data.generic import GenericData
            import importlib.resources as pkg_resources
            import spotRiver.data as data
            inp_file = pkg_resources.files(data)
            csv_path = str(inp_file.resolve())
            dataset = GenericData(filename="UnivariateData.csv",
                                directory=csv_path,
                                target="Consumption",
                                n_features=1,
                                n_samples=51_706,
                                converters={"Consumption": float},
                                parse_dates={"Time": "%Y-%m-%d %H:%M:%S%z"})
            for x, y in dataset:
                print(x, y)
                break
            {'Time': datetime.datetime(2016, 12, 31, 23, 0, tzinfo=datetime.timezone.utc)} 10951.217

    """
    def __init__(
        self,
        filename: str,
        target: str,
        n_features: int,
        n_samples: int,
        converters: Dict[str, callable],
        parse_dates: List[str],
        directory: str,
        task: str = base.REG,
        fraction: float = 1.0
    ):
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

    def __iter__(self) -> Union[Dict[str, float], float]:
        return stream.iter_csv(
            self.path,
            target=self.target,
            converters=self.converters,
            parse_dates=self.parse_dates,
            fraction=self.fraction,
            seed=123,
        )