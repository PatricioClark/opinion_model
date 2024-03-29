""" PINN implementation of opinion model """

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import opinion_model

import numpy as np

lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.9)
layers  = [2] + 2*[64] + [1]
PINN = PhysicsInformedNN(layers,
                         dest='./odir/',
                         activation='elu',
                         optimizer=keras.optimizers.Adam(lr),
                         restore=False)
PINN.model.summary()


Lx = 2
Nx = 1000
# X = [(t_0, x_0), (t_1, x_0), ...]
# Y = [u(t_0, x_0), u(t_1, x_0), ...]
# lambda_data = [1, 0, 0, ...]
# lambda_phys = [0, 1, 1, ...]
# flags = [0, 1, 2, ...]

# Add validation function
def cte_validation(self, X, Nt, Nx, u):
    # Definimos una función que el código después llama
    # El único parametro que le pasa código es el número de epoch
    # El resto lo definimos al generar la función
    def validation(ep):
        # Get prediction
        Y  = self.model(X)[0].numpy()

        u_p = Y[:,0].reshape((Nt,Nx))

        # True valu
        u_t  = u(X)

        # Error global
        err  = np.sqrt(np.mean((u_p-u_t)**2))/np.std(u_t)

        # Loss functions
        output_file = open(self.dest + 'validation.dat', 'a')
        print(ep, *err,
              file=output_file)
        output_file.close()

    return validation

# Se la agregamos a la clase
PINN.validation = cte_validation(PINN, X, Nt, Nx, u)

alpha   = 0.0
tot_eps = 4000
eq_params = [Lx/Nx]
eq_params = [np.float32(p) for p in eq_params]
PINN.train(X, Y, opinion_model,
           epochs=tot_eps,
           eq_params=eq_params,
           batch_size=Nx,
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,   # Punto donde se enfuerza L_pde
           flags=flags,               # Separa el dataset a cada t
           rnd_order_training=False,  # No arma batches al hacer
           alpha=alpha,
           verbose=False,
           valid_freq=10,             # Cada cuantas epochs usa la funcion validation
           timer=False)
