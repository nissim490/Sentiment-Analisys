_THRESHOLD_NEGATIVE = 0.4
_THRESHOLD_POSITIVE = 0.6


def labels():
    """
    Gets the textual representation of labels by indexing order.
    Where 0 -> negative, 1 -> positive and 2 -> neutral.
    This ordinal order keeps with the integer representation
    of the labels in our datasets.
    :return: list of labels
    """
    return ['Negative', 'Positive', 'Neutral']


def predication_to_label(predication):
    """
    Converts a prediction result into a textual representation
    of the result.
    The range of the prediction is separated into areas which match the labels.
    :param predication: prediction result, from 0 to 1
    :return: textual representation
    """
    text_result = 'Positive'
    if predication < _THRESHOLD_NEGATIVE:
        text_result = 'Negative'
    elif predication < _THRESHOLD_POSITIVE:
        text_result = 'Neutral'

    return text_result
