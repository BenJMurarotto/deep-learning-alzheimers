import numpy as np
from tensorflow import keras
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

def make_mlp(input_shape, layers, units, activation):
    model = keras.Sequential()
    model.add(
        keras.layers.InputLayer(
            shape=(np.prod(input_shape),)
        )
    )
    for i in range(layers):
        model.add(keras.layers.Dense(units=units, activation=activation))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def calculate_accuracy(predictions, labels):
    return (
        accuracy_score(predictions, labels)
    )

def map_estimate(predictions):
    return (predictions[:, 1] >= 0.5).astype(int).reshape(-1)

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} of training; {} : {}".format(epoch, logs.keys(), logs.values()), end='\r')

earlyStopping = keras.callbacks.EarlyStopping(
    monitor="accuracy",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)


class MLP(BaseEstimator):
    def __init__(
        self,
        layers=3,
        units=100,
        grid_size=50,
        spline_order=5,
        activation='gelu',
    ):
        self.layers = layers
        self.units = units
        self.activation = activation
        
    def get_params(self, deep=True):
        return {
            'layers': self.layers,
            'units': self.units,
            'activation': self.activation,
        }
        
    def fit(self, inputs, targets):
        self.input_shape = (50, 50, 1)
        targets = targets.reshape(-1, 1)
        targets = np.hstack([targets, targets])
        targets[:, 0] = 1-targets[:, 1]
        self.model = make_mlp(self.input_shape, self.layers, self.units, self.activation)
        self.model.fit(
            inputs.reshape(-1, np.prod(self.input_shape)),
            targets,
            epochs=20,
            callbacks=[CustomCallback(), earlyStopping],
            verbose=0,
        )
    
    def score(self, inputs, targets):
        predictions = self.model.predict(inputs.reshape(-1, np.prod(self.input_shape)))
        map_estimates = map_estimate(predictions)
        return calculate_accuracy(map_estimates, targets)
