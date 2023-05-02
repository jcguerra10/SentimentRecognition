from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, Input, Dropout

def model(MAX_LEN):
        """
        Defines a sequential Keras model for text classification using an LSTM layer with dropout and dense layers.

        :param MAX_LEN: Maximum length of input sequences.
        :type MAX_LEN: int
        :return: A Keras model instance.
        :rtype: keras.Model

        Example usage:
        >>> model = model(MAX_LEN=100)

        Model architecture:
        - Input layer with shape (MAX_LEN, 50)
        - LSTM layer with 64 units
        - Dropout layer with rate 0.2 to prevent overfitting
        - Flatten layer to convert the LSTM output to a 1D tensor
        - Dense layer with 128 units and ReLU activation function
        - Output layer with 7 units and softmax activation function

        """
        lstm_dropout_dense = Sequential(name='Lstm-dout-dense')
        lstm_dropout_dense.add(Input(shape=(MAX_LEN, 50)))
        lstm_dropout_dense.add(LSTM(64))
        lstm_dropout_dense.add(Dropout(0.2))
        lstm_dropout_dense.add(Flatten())
        lstm_dropout_dense.add(Dense(128, activation='relu'))
        lstm_dropout_dense.add(Dense(7, activation='softmax'))
        return lstm_dropout_dense