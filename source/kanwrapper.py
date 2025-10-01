import numpy as np
from tensorflow import keras
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score

from keras_efficient_kan import KANLinear

def make_kan(input_shape, layers, units, grid_size, spline_order, activation):
    model = keras.Sequential()
    model.add(
        keras.layers.InputLayer(
            shape=(np.prod(input_shape),)
        )
    )
    for i in range(layers):
        model.add(KANLinear(units=units, grid_size=grid_size, spline_order=spline_order, base_activation=activation))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['auc'])
    return model

def calculate_auc(predictions, labels):
    return (
        roc_auc_score(predictions, labels)
    )

def map_estimate(predictions):
    return (predictions[:, 1] >= 0.5).astype(int).reshape(-1)

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} of training; {} : {}".format(epoch, logs.keys(), logs.values()), end='\r')

earlyStopping = keras.callbacks.EarlyStopping(
    monitor="auc",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)


class KAN(BaseEstimator):
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
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.activation = activation
        
    def get_params(self, deep=True):
        return {
            'layers': self.layers,
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'activation': self.activation,
        }
        
    def fit(self, inputs, targets):
        self.input_shape = (50, 50, 1)
        targets = targets.reshape(-1, 1)
        targets = np.hstack([targets, targets])
        targets[:, 0] = 1-targets[:, 1]
        self.model = make_kan(self.input_shape, self.layers, self.units, self.grid_size, self.spline_order, self.activation)
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
        return calculate_auc(map_estimates, targets)
