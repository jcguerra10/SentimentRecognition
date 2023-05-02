from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, Input, Dropout

def model(MAX_LEN):
        """
        Defines a sequential Keras model for text classification using three LSTM layers, a CNN layer, and dense layers.

        :param MAX_LEN: Maximum length of input sequences.
        :type MAX_LEN: int
        :return: A Keras model instance.
        :rtype: keras.Model

        Example usage:
        >>> model = model(MAX_LEN=100)

        Model architecture:
        - Input layer with shape (MAX_LEN, 50)
        - Three LSTM layers with 256, 128, and 64 units respectively, all with dropout rate 0.2
        - 1D convolutional layer with 128 filters, kernel size 3, and ReLU activation function
        - Max pooling layer with pool size 3 and strides 2
        - Dropout layer with rate 0.2 to prevent overfitting
        - Flatten layer to convert the CNN output to a 1D tensor
        - Dense layer with 64 units and ReLU activation function
        - Output layer with 7 units and softmax activation function

        """
        lstm3_do_cnn_dense = Sequential(name='3LSTM-DO-CNN-Dense')
        lstm3_do_cnn_dense.add(Input(shape=(MAX_LEN, 50)))

        #LSTM
        lstm3_do_cnn_dense.add(LSTM(256, name='LSTM1', return_sequences=True))
        lstm3_do_cnn_dense.add(Dropout(0.2, name='DO1'))

        lstm3_do_cnn_dense.add(LSTM(128, name='LSTM2', return_sequences=True))
        lstm3_do_cnn_dense.add(Dropout(0.2, name='DO2'))

        lstm3_do_cnn_dense.add(LSTM(64, name='LSTM3', return_sequences=True))
        lstm3_do_cnn_dense.add(Dropout(0.2, name='DO3'))

        #CNN
        lstm3_do_cnn_dense.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
        lstm3_do_cnn_dense.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
        lstm3_do_cnn_dense.add(Dropout(0.2))
        lstm3_do_cnn_dense.add(Flatten(name='F1'))

        #Fully connected
        lstm3_do_cnn_dense.add(Dense(64, activation='relu'))
        lstm3_do_cnn_dense.add(Dense(7, activation='softmax'))
        return lstm3_do_cnn_dense

