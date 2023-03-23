""" PINN implementation of opinion model """

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import opinion_model

import numpy as np

# Cargamos la red de vuelta
# Importante que los hyperparams coincidan y que restore=True
lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.9)
layers  = [2] + 2*[64] + [1]
PINN = PhysicsInformedNN(layers,
                         dest='./', # Mismo directorio que el dest de run_pinn.py
                         activation='elu',
                         optimizer=keras.optimizers.Adam(lr),
                         restore=True)

# Defino los puntos donde evaluar
# Evaluar en los puntos de entrenamiento es buena idea
# pero puede ser en cualquier punto del dominio
# X = ...
Y_obs = PINN.model(X)
