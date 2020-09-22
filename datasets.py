import pandas

_TRAIN_DATA_PATH = 'data/training.1600000.processed.noemoticon.csv'
_TEST_DATA_PATH = 'data/testdata.manual.2009.06.14.csv'
_COLUMNS = ["sentiment", "id", "date", "query", "user", "text"]


def load_train_data():
    data = pandas.read_csv(
        _TRAIN_DATA_PATH,
        header=None,
        names=_COLUMNS,
        engine="python",
        encoding="latin1"
    )

    data.drop(["id", "date", "query", "user"],
              axis=1,
              inplace=True)

    data_labels = data.sentiment.values
    data_labels[data_labels == 4] = 1

    return data.text, data_labels


def load_test_data():
    data = pandas.read_csv(
        _TEST_DATA_PATH,
        header=None,
        names=_COLUMNS,
        engine="python",
        encoding="latin1"
    )

    data.drop(["id", "date", "query", "user"],
              axis=1,
              inplace=True)

    data_labels = data.sentiment.values
    data_labels[data_labels == 4] = 1

    return data.text, data_labels
