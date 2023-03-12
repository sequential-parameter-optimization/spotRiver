from river import datasets
import pandas as pd


def convert_to_df(dataset: datasets.base.Dataset, target_column: str) -> pd.DataFrame:
    """Converts a river dataset into a pandas DataFrame.

    Args:
        dataset (datasets.base.Dataset): The river dataset to be converted.
        target_column (str): The name of the target column in the resulting DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame representation of the given dataset.

    Example:
        >>> dataset = datasets.TrumpApproval()
            target_column = "Approval"
            df = convert_to_df(dataset, target_column)
            df.rename(columns={
                'date': 'ordinal_date',
                'Gallup': 'gallup',
                'Ipsos': 'ipsos',
                'Morning Consult': 'morning_consult',
                'Rasmussen': 'rasmussen',
                'YouGov': 'you_gov'},
                inplace=True)
            # Split the data into train and test sets
            train = df[:500]
            test = df[500:]
    """
    data_dict = {key: [] for key in list(dataset)[0][0].keys()}
    data_dict[target_column] = []
    for x in dataset:
        for key, value in x[0].items():
            data_dict[key].append(value)
        data_dict[target_column].append(x[1])
    df = pd.DataFrame(data_dict)
    return df
