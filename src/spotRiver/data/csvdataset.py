import pandas as pd
import pathlib


class CSVDataset:
    """
    A Dataset for handling CSV data.

    Args:
        filename (str): The path to the CSV file. Defaults to "data.csv".
        directory (str): The path to the directory where the CSV file is stored. Defaults to None.
        feature_type (dtype): The data type of the features. Defaults to float.
        target_column (str): The name of the target column. Defaults to "y".
        target_type (dtype): The data type of the targets. Defaults to float.
        train (bool): Whether the dataset is for training or not. Defaults to True.
        rmNA (bool): Whether to remove rows with NA values or not. Defaults to True.
        dropId (bool): Whether to drop the "id" column or not. Defaults to False.
        **desc (Any): Additional keyword arguments.

    Examples:
        >>> from spotRiver.data.csvdataset import CSVDataset
            dataset = CSVDataset(csv_file='data.csv', target_column='prognosis', feature_type=int)
            print(dataset.data.shape)
    """

    def __init__(
        self,
        filename: str = None,
        directory: None = None,
        feature_type: type = float,
        target_column: str = "y",
        target_type: type = float,
        train: bool = True,
        rmNA=True,
        dropId=False,
        **desc,
    ) -> None:
        # super().__init__()
        self.filename = filename
        self.directory = directory
        self.feature_type = feature_type
        self.target_type = target_type
        self.target_column = target_column
        self.train = train
        self.rmNA = rmNA
        self.dropId = dropId
        self.data = self._load_data()

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content

    def _load_data(self) -> tuple:
        # print(f"Loading data from {self.path}")
        df = pd.read_csv(self.path, index_col=False)
        # rm rows with NA
        if self.rmNA:
            df = df.dropna()
        if self.dropId:
            df = df.drop(columns=["id"])
        return df
