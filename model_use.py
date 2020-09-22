import pickle

import model as modelim

_SAVE_PATH = 'save/model.h5'
_TOKENIZER_PATH = 'save/tokenizer.pickle'


def create(tokenizer):
    """
    Creates the model.
    :param tokenizer: tokenizer for words
    :return: compiled untrained model
    """
    model = modelim.create_cnn_model(tokenizer.vocab_size)
    modelim.compile_model(model)
    return model


def load_model():
    """
    Loads model from the file system.
    :return: loaded model, tokenizer
    """
    with open(_TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    model = modelim.create_cnn_model(tokenizer.vocab_size)
    model.load_weights(_SAVE_PATH)
    modelim.compile_model(model)
    return model, tokenizer


def save_model(model, tokenizer):
    """
    Saves model into the file system.
    :param model: model to save
    :param tokenizer: tokenizer for words known to the model
    """
    model.save_weights(_SAVE_PATH)

    with open(_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
