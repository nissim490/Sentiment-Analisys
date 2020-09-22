import re

import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences


def _clean_tweet(tweet):
    """
    Performs first level of cleaning on the tweet text.
    Remove HTML tokens, user mentions (@) and URL links.
    :param tweet: tweet text
    :return: cleaned tweet
    """
    # it can also parse through our data and get text foreach line.
    tweet = BeautifulSoup(tweet, "html.parser").get_text()
    # Removing the @
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Removing the URL links
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
    return tweet


def _clean_tweet_2(tweet):
    """
    Performs second level of cleaning on the tweet text.
    Remove non letter characters and additional whitespaces.
    :param tweet: tweet text
    :return: cleaned tweet
    """

    # Keeping only letters
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Removing additional whitespaces
    tweet = re.sub(r" +", ' ', tweet)
    return tweet


def clean_data(texts, in_between_cleaning=None):
    """
    before removing non-letter characters.
    :param texts: texts to process
    :param in_between_cleaning: function to run on the texts, or none to skip this part
    :return: processed texts, processed labels, and result from in_between_cleaning, if not None
    """

    cleaned_data = [_clean_tweet(tweet) for tweet in texts]
    if in_between_cleaning is not None:
        in_between_cleaning_results = in_between_cleaning(cleaned_data)
    else:
        in_between_cleaning_results = None
    cleaned_data = [_clean_tweet_2(tweet) for tweet in cleaned_data]
    return cleaned_data, in_between_cleaning_results


def create_tokenizer(data_clean):
    """
    creates a tokenizer from a given text. Uses this text as a vocabulary of words.
    :param data_clean: list of texts which provide a vocabulary
    :return: tokenizer, for text encoding
    """

    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        data_clean, target_vocab_size=2 ** 17 #the maximum size of the vocabulary is 2**17
    )


def tokenize(data_clean, tokenizer):
    """
    encodes a list of data using a tokenizer.
    :param data_clean: data to encode
    :param tokenizer: tokenizer to use for encoding
    :return: list of encoded data
    """
    return [tokenizer.encode(sentence) for sentence in data_clean]


def pad_data(data_inputs, padding_size=None):
    """
    pad the data so that each row is the same size.
    pads with '0' data after the actual data.
    :param data_inputs: data after being tokenized
    :param padding_size: size to pad to, or None to calculate it automatically from the
    largest vector in data
    :return: the padded data
    """

    if padding_size is None:
        padding_size = max([len(sentence) for sentence in data_inputs])

    return pad_sequences(data_inputs,
                         value=0,
                         padding="post",
                         maxlen=padding_size)


def preprocess(texts, in_between_cleaning=None, tokenizer=None, padding_size=None):
    """
    preprocess method Performs preprocessing on given texts.
    :param texts: list of texts to process
    :param in_between_cleaning: function to run on the text data before removing all non-letter characters
    :param tokenizer: tokenizer to use for encoding, or if None, builds a new tokenizer for the given texts
    :param padding_size: size to pad to, or None to calculate one automatically
    :return: transformed data, tokenizer, result of in_between_cleaning if not None
    """
    texts, in_between_cleaning_results = clean_data(texts, in_between_cleaning)

    #create the tokenizer
    if tokenizer is None:
        tokenizer = create_tokenizer(texts)

    #encode the tokenizer
    texts = [tokenizer.encode(text) for text in texts]
    #padding all of the sentences into the maximum size of sentence in the data
    return pad_data(texts, padding_size), tokenizer, in_between_cleaning_results
