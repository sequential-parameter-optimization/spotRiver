from spotRiver.data.selector import data_selector
from spotRiver.utils.data_conversion import convert_to_df


def test_data_selector():
    data_set_values = [
        "Bananas",
        "Elec2",
        "HTTP",
        "Phishing",
    ]
    for i in range(len(data_set_values)):
        data_set = data_set_values[i]
        dataset, n_samples = data_selector(
            data_set=data_set
        )
        df = convert_to_df(dataset, target_column="y", n_total=None)
        assert df.shape[0] == n_samples