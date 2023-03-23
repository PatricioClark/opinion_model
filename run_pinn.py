""" PINN implementation of opinion model """

import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

from pinn      import PhysicsInformedNN
from equations import opinion_model

import numpy as np

lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.9)
layers  = [2] + 2*[64] + [1]
PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='elu',
                         optimizer=keras.optimizers.Adam(lr),
                         restore=False)
PINN.model.summary()

def u(x):  
  lower = x[0] - 1
  uper = 1 - x[0]
  pos = np.delete(x,0)
  sol = np.where((pos<lower) | (pos>uper),0,1) * 1/(2-2*x[0])      
  sol = np.insert(sol,0,x[0])
  return sol


Lx = 2
Nx = 100
Nt = 500

t = np.linspace(0,100,Nt)
times = np.repeat(t,Nx)
arr = np.array(np.array_split(times,len(times))).flatten()
arr = np.insert(arr,np.arange(1,Nx*Nt+1),(2*np.random.rand(1,Nx*Nt) - 1).flatten())
X = np.array(np.split(arr,Nt*Nx)).reshape(Nx*Nt,2)
X = X

#X = np.array(2*np.random.rand(Nt,Nx+1) - 1)
#for i in range(len(t)):
#  X[i][0] = t[i]

Y =np.array([u(x) for x in X]) #[u(t_0,x_0),u(t_1,x_1),...]
Y = Y
lambda_data = np.zeros(Nt) #[1,0,0,..]
lambda_data[0] = 1
lambda_phys = np.ones(Nt)
lambda_phys[0] = 0 #[0,1,1,..]
flags = np.arange(Nt) 

alpha   = 0.0
tot_eps = 10
eq_params = [Lx/Nx]
eq_params = [np.float32(p) for p in eq_params] #eq_params es el diferencial de x que paso para calcular la integral
PINN.train(X, Y, opinion_model,
           epochs=tot_eps,
           eq_params=eq_params,
           batch_size=Nt*Nx,
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,   # Punto donde se enfuerza L_pde
           flags=flags,               # Separa el dataset a cada t
           rnd_order_training=False,  # No arma batches al hacer
           alpha=alpha,
           verbose=True,
           timer=False)

