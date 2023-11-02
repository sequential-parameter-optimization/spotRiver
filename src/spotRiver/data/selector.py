from river import datasets


def data_selector(data_set):
    """
    Selects the data set to be used.

    Args:
        data_set (str, optional):
            Data set to use. Defaults to "Phishing".

    Returns:
        dataset (object):
            Data set to use.
        n_samples (int):
            Number of samples in the data set.
    Raises:
        ValueError:
            If data_set is not "Bananas" or "CreditCard" or "Elec2" or "Higgs" or
            "HTTP" or "MaliciousURL" or "Phishing" or "SMSSpam" or "SMTP" or "TREC07".

    Examples:
        >>> dataset, n_samples = data_selector("Phishing")


    """
    if data_set == "Bananas":
        dataset = datasets.Bananas()
        n_samples = 5300
    elif data_set == "CreditCard":
        dataset = datasets.CreditCard()
        n_samples = 284_807
    elif data_set == "Elec2":
        dataset = datasets.Elec2()
        n_samples = 45_312
    elif data_set == "Higgs":
        dataset = datasets.Higgs()
        n_samples = 11_000_000
    elif data_set == "HTTP":
        dataset = datasets.HTTP()
        n_samples = 567_498
    elif data_set == "MaliciousURL":
        dataset = datasets.MaliciousURL()
        n_samples = 2_396_130
    elif data_set == "Phishing":
        dataset = datasets.Phishing()
        n_samples = 1250
    elif data_set == "SMSSpam":
        dataset = datasets.SMSSpam()
        n_samples = 5574
    elif data_set == "SMTP":
        dataset = datasets.SMTP()
        n_samples = 95_156
    elif data_set == "TREC07":
        n_samples = 75_419
        dataset = datasets.TREC07()
    else:
        raise ValueError(
            'data_set must be "Bananas" or "CreditCard" or "Elec2" or "Higgs" or "HTTP" or "MaliciousURL" or "Phishing" or "SMSSpam" or "SMTP" or "TREC07"'
        )
    return dataset, n_samples
