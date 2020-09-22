import re

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

# regex modified to positive/negative
REGEX_NEGATIVE_EMOJI = r':\(|:-\(|:-O|:\||:S|:\$|:@|8o\||\+o\('
REGEX_POSITIVE_EMOJI = r':\)|:-\)|;\);-\)|:P|:D|:S|\(H\)C\)'

CHECKPOINT_SAVE_PREFIX = 'save/model-checkpoint'


def weak_data_matcher(texts):
    """
    Extracts positive and negative tweets from texts.
    :param texts: texts to extracts from
    :return: tuple of indices for tweets with positive, negative alignments respectively
    """
    # Emoticon alignment inferring,
    # In the article (page 3) Distant Supervised Phase:
    # The label is inferred by the emoticons inside the tweet, where we ignore tweets with opposite emoticons.

    pos_re = re.compile(REGEX_POSITIVE_EMOJI)
    neg_re = re.compile(REGEX_NEGATIVE_EMOJI)
    pos_matches = []
    neg_matches = []
    for i in range(len(texts)):
        line = texts[i]
        match_pos = pos_re.search(line) #checks for a match anywhere in the string
        match_neg = neg_re.search(line)
        if match_pos and not match_neg:
            pos_matches.append(i)
        elif match_neg and not match_pos:
            neg_matches.append(i)

    return pos_matches, neg_matches


def _split_data_for_weak_train(x, y, weak_data):
    """
    Splits the given dataset into datasets for weak-supervised phase and
    supervised phase.
    :param x: pre-processed tweet data
    :param y: labels
    :param weak_data: tuple of categorized indices for data to collect for weak-supervised phase.
    :return: tuple of
        - weak supervised positive tweets
        - weak supervised negative tweets
        - supervised tweets
        - supervised labels
    """
    pos_indices, neg_indices = weak_data
    positive = np.row_stack([x[i] for i in pos_indices])
    negative = np.row_stack([x[i] for i in neg_indices])
    x_new = np.delete(x, pos_indices + neg_indices, axis=0)
    y_new = np.delete(y, pos_indices + neg_indices, axis=0)
    return positive, negative, x_new, y_new


def _weak_supervised_train(model, x_positive, x_negative, batch_size):
    """
    Performs the semi-supervised training phase.
    We train for one epoch on this set like in the article
    :param model: model to train
    :param x_positive: pre-processed tweets which are inferred to be positive
    :param x_negative: pre-processed tweets which are inferred to be negative
    :param batch_size: size of training data batches
    :return: results of the training
    """
    y_positive = np.ones(len(x_positive))#create the labels for the week data 0/1
    y_negative = np.zeros(len(x_negative))

    x = np.concatenate((x_positive, x_negative))#Concatenate the tweets
    y = np.concatenate((y_positive, y_negative))#Concatenate the labels

    shuffler = np.random.permutation(len(x))#define the shuffler
    x = x[shuffler]#shuffler the tweets and label match to each tweet
    y = y[shuffler]#the label is suffler respectively to the tweet

    # distant-supervised training,
    # In the article, page 3, Distant Supervised Phase:
    # We pre-train the CNN for 1 epoch on an weakly labelled dataset
    history = model.fit(x, y,
                        epochs=1,
                        batch_size=batch_size,
                        verbose=1)

    return history


def _supervised_train(model, x, y, batch_size, num_epoch):
    """
    Performs the supervised training phase.
    :param model: model to train
    :param x: pre-processed tweet data
    :param y: labels
    :param batch_size: size of training data batches
    :param num_epoch: amount of epochs to run
    :return: list of results of the training
    """
    # early stopping,
    # In the article (page 3) Supervised Phase:
    # In each round we train the CNN using early stopping on the held-out set
    early_stopping = EarlyStopping(monitor='val_loss', mode='min',
                                   patience=5,
                                   restore_best_weights=True,
                                   verbose=1)

    # actually train
    history = model.fit(x, y,
                        epochs=num_epoch,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stopping])

    return history


def train(model, x, y, weak_data, batch_size, num_epoch, data_limit=None):
    """
    Trains the given model using the training method specified in the article.
    The training is made up of 2 phases.
    :param model: model to train
    :param x: pre-processed tweet text
    :param y: labels
    :param weak_data: tuple of: indices for positive-inferred tweets, indices for negative-inferred tweets
    :param batch_size: size of training data batches
    :param num_epoch: amount of epochs to run in supervised phase
    :param data_limit: limits the amount of data used in supervised phase, or None to use all
    :return: tuple of weakly-supervised, supervised training results
    """
    weak_pos, weak_neg, x, y = _split_data_for_weak_train(x, y, weak_data)
    if data_limit:# if the data_limit is not null
        x = x[:data_limit]
        y = y[:data_limit]

    weak_results = _weak_supervised_train(model, weak_pos, weak_neg, batch_size)
    supervised_results = _supervised_train(model, x, y, batch_size, num_epoch)

    return weak_results, supervised_results


def validate(model, x, y, num_epoch, n_splits=2):
    """
    Validates the model.
    Uses a K-fold cross validation training on the given set of data.
    :param model: model to train
    :param x: pre-processed tweet data
    :param y: labels
    :param num_epoch: amount of epochs to run
    :param n_splits: K constant for the K-fold cross validation
    :return: list of results of the training and evaluation for each fold
    """
    results = []
    checkpoint = tf.train.Checkpoint(model=model)#
    # Save the model status
    checkpoint_path = checkpoint.save(CHECKPOINT_SAVE_PREFIX)

    # K-fold cross validation,
    # In our case K=2
    for train_index, test_index in KFold(n_splits).split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        history = model.fit(x_train, y_train,
                            epochs=num_epoch,
                            verbose=1)
        # evaluate
        evaluation = model.evaluate(x_test, y_test)

        results.append((history, evaluation))

        # Restore changes, we don't want to modify the model, we simply check it
        checkpoint.restore(checkpoint_path)

    return results
