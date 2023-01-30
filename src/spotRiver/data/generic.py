from river import stream

from . import base


class GenericData(base.GenericFileDataset):
    """Generic File Data Class

    Args:
        base (_type_): _description_
    """
    def __init__(self, filename, target, n_features, n_samples, converters, parse_dates, directory, task=base.REG):
        """Generic File Data

        Args:
            filename (_type_): _description_
            target (_type_): _description_
            n_features (_type_): _description_
            n_samples (_type_): _description_
            converters (_type_): _description_
            parse_dates (_type_): _description_
            directory:
            task (_type_, optional): _description_. Defaults to base.REG.
        """
        super().__init__(
            filename=filename,
            n_features=n_features,
            n_samples=n_samples,
            task=task,
            target=target,
            converters=converters,
            parse_dates=parse_dates,
            directory=directory
        )

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target=self.target,
            converters=self.converters,
            parse_dates=self.parse_dates
        )
