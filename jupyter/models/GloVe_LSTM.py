from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, Input

def model(MAX_LEN):
        """
        Defines a sequential Keras model for text classification using an LSTM layer.

        :param MAX_LEN: Maximum length of input sequences.
        :type MAX_LEN: int
        :return: A Keras model instance.
        :rtype: keras.Model

        Example usage:
        >>> model = model(MAX_LEN=100)

        Model architecture:
        - Input layer with shape (MAX_LEN, 50)
        - LSTM layer with 64 units
        - Flatten layer to convert the LSTM output to a 1D tensor
        - Dense output layer with 7 units and softmax activation function

        """
        model = Sequential()
        model.add(Input(shape=(MAX_LEN, 50)))
        model.add(LSTM(64))
        model.add(Flatten())
        model.add(Dense(7, activation='softmax'))
        return model