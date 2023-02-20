from spotRiver import data
from spotRiver.utils.features import get_hour_distances, get_month_distances, get_ordinal_date, get_weekday_distances


def test_features():
    """
    Test features
    """
    dataset = data.AirlinePassengers()

    for x, _ in dataset:
        assert(get_ordinal_date(x)["ordinal_date"] ==  711493)
        assert(get_hour_distances(x)["0"] == 1.0)
        assert(get_weekday_distances(x)["Saturday"] == 1.0)
        assert(get_month_distances(x)["January"] == 1.0)
        break