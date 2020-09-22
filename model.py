import tensorflow as tf
from tensorflow.keras import layers


def create_cnn_model(vocab_size, h=5, s=3, strides=2, m=200, embed_size=128, p=0.2):
    """
    :param vocab_size: size of the vocabulary known (word count)
    :param h: filter window size for the convolutional layers
    :param s: pooling size for the first max-pooling layer
    :param strides: strides for the first max-pooling layer
    :param m: filter count
    :param embed_size: size of the embedding vector
    :param p: dropout probability
    :return: built uncompiled model
    """
    model = tf.keras.Sequential()
    # embedding
    # The first layer of the network consists of a lookup table where the word embeddings are represented
    # mention in the article (page 3) Word Embeddings: we create the word embeddings in phase P 1 using word2vec.
    model.add(layers.Embedding(vocab_size, embed_size))

    # dropout
    # Due to overfitting, this layer was added after input layer, with drop rate 0.2
    # In the article, page 2, Dropout:
    # Dropout is an alternative technique used to reduce overfitting
    # We apply Dropout to the hidden layer and to the input layer using p = 0.2 in both cases
    model.add(layers.Dropout(rate=p))

    # first convolutional layer
    # - About filter parameters and window size
    # In the article (page 2) Convolutional layer:
    # This layer, have a set of m filters is applied to a sliding window of length h over each sentence
    # In the article (page 2) parameters:
    # For both convolutional layers we set the length of the sliding window h to 5.
    # And the number of filters m is set to 200 in both convolutional layers.
    # The activation function in this layer is relu
    # The output of the convolutional layer is passed through a non-linear activation function
    model.add(layers.Conv1D(filters=m, kernel_size=h, activation=tf.nn.relu))

    #Max pooling
    # In the article (page 2) Max pooling:
    # Where s is the length of each interval. In the case of overlapping intervals with a stride value
    # In the article (page 2) parameters:
    # pooling interval s is set to 3 in both layers, where we use a striding of 2 in the first layer
    model.add(layers.MaxPool1D(pool_size=s, strides=strides))
    # second convolutional layer
    # similar to first layer
    model.add(layers.Conv1D(filters=m, kernel_size=h, activation=tf.nn.relu))
    # this time we use a global max-pooling to reduce the data

    model.add(layers.GlobalMaxPooling1D())
    # hidden
    # - About the type of layer
    # In the article (page 2) Hidden layer:
    # A fully connected hidden layer computes the transformation
    # In the article (page 2) Hidden layer:
    # We use with rectified linear (relu) activation function
    # - About the node units
    # the number of filters m is set to 200 in both convolutional layers.
    model.add(layers.Dense(m, activation=tf.nn.relu))
    # dropout
    # We apply Dropout to the hidden layer and to the input layer using p = 0.2 in both cases
    model.add(layers.Dropout(rate=p))
    # dense
    # using sigmoid function
    # In the article (page 2) they use Softmax amd we use sigmoid function:
    # Finally, the outputs of the hidden layer
    model.add(layers.Dense(1, activation=tf.nn.sigmoid))
    return model


def compile_model(model):
    # Compile the model before training

    model.compile(loss="binary_crossentropy", #its the bias of the prediction
                  optimizer="adam",
                  metrics=["accuracy"])
