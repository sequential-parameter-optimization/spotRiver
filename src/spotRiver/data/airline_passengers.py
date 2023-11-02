from river import stream

from . import base


class AirlinePassengers(base.FileDataset):
    """Monthly number of international airline passengers [1].

    The stream contains 144 items and only one single feature, which is the month. The goal is to
    predict the number of passengers each month by capturing the trend and the seasonality of the
    data.

    Returns:
        (Generator): An iterator over the data in the file.

    Note: The code can be used as a template for creating new datasets based on CSV files.

    Examples:
        >>> from spotRiver.data.airline_passengers import AirlinePassengers
            dataset = AirlinePassengers()
            for x, y in dataset.take(5):
                print(x, y)
            {'month': datetime.datetime(1949, 1, 1, 0, 0)} 112
            {'month': datetime.datetime(1949, 2, 1, 0, 0)} 118
            {'month': datetime.datetime(1949, 3, 1, 0, 0)} 132
            {'month': datetime.datetime(1949, 4, 1, 0, 0)} 129
            {'month': datetime.datetime(1949, 5, 1, 0, 0)} 121

    References:
        International airline passengers: monthly totals in thousands. Jan 49 – Dec 60
    """

    def __init__(self):
        """Constructor method.

        Returns:
            (NoneType): None

        """
        super().__init__(
            filename="airline-passengers.csv",
            task=base.REG,
            n_features=1,
            n_samples=144,
        )

    def __iter__(self):
        """Iterate over the data.
        Returns:
            (Generator): An iterator over the data in the file.
        """
        return stream.iter_csv(
            self.path,
            target="passengers",
            converters={"passengers": int},
            parse_dates={"month": "%Y-%m"},
        )
