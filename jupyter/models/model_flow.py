from keras.callbacks import ModelCheckpoint, EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)
cp = ModelCheckpoint('saved/', save_best_only=True)

def compile(model):
        """
        Compiles a Keras model with the Adam optimizer, categorical cross-entropy loss function, and accuracy metrics.

        :param model: A Keras model instance to be compiled.
        :type model: keras.Model
        :return: The compiled Keras model instance.
        :rtype: keras.Model

        Example usage:
        >>> model = compile(model)

        """
        model.compile(optimizer='adam',
                loss='categorical_crossentropy',
              metrics=['acc'])
        return model
        
def train(model,X_train,y_train):
        """
        Trains a Keras model on input data and labels.

        :param model: A Keras model instance to be trained.
        :type model: keras.Model
        :param X_train: Input data for training.
        :type X_train: numpy.ndarray
        :param y_train: Labels for training data.
        :type y_train: numpy.ndarray
        :return: The training history of the Keras model.
        :rtype: keras.callbacks.History

        Example usage:
        >>> history = train(model, X_train, y_train)

        """
        history = model.fit(X_train, y_train,
                    validation_split=0.1, epochs=10,
                    callbacks=[cp, early_stop])
        return history

