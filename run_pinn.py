""" PINN implementation of opinion model """

import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

from pinn      import PhysicsInformedNN
from equations import opinion_model
import matplotlib.pyplot as plt
import numpy as np

#lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.9)
lr = 1e-3
layers  = [2] + 3*[64] + [1]
PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='elu',
                         optimizer=keras.optimizers.Adam(lr),
                         restore=True)
PINN.model.summary()


def u(X):
  sol = np.zeros(len(X))
  for i in range(len(X)):
    x = X[i,1]
    t = X[i,0]
    lower = t - 1
    uper = 1 - t
    sol[i] = np.where((x<lower) | (x>uper),0,1) * 1/(2-2*t)        
  return sol.reshape((len(X),1))

Lx = 2
Nx = 100
Nt = 10 #Hay un error cuando Nx > Nt para armar los flags

t = np.linspace(0,100,Nt)
x = np.linspace(-1,1,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)
Y = u(X) #[u(t_0,x_0),u(t_1,x_1),...]

lambda_data = np.zeros(Nt*Nx) #[1,0,0,..]
lambda_data[:Nx] = 1
lambda_phys = np.ones(Nt*Nx)
lambda_phys[:Nx] = 0 #[0,1,1,..]
flags = np.repeat(np.arange(Nt),Nx)

alpha   = 0.0
tot_eps = 300
eq_params = [Lx/Nx]
eq_params = [np.float32(p) for p in eq_params] #eq_params es el diferencial de x que paso para calcular la integral

PINN.train(X, Y, opinion_model,
           epochs=tot_eps,
           eq_params=eq_params,
           batch_size=Nt,
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,   # Punto donde se enfuerza L_pde
           flags=flags,               # Separa el dataset a cada t
           rnd_order_training=False,  # No arma batches al hacer
           alpha=alpha,
           verbose=True,
           timer=False)



fields = (PINN.model(X)[0]).numpy()


fig = plt.figure(figsize = (10,10))
plt.title('Model')
ax = plt.axes(projection='3d')
ax.scatter3D(X[:Nx,0].flatten(), X[:Nx,1].flatten(), fields[:Nx])
#ax.view_init(10,90)
plt.show()



