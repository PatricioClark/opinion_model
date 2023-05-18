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


def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

def solution(X):  
  sol = np.zeros(len(X))
  for i in range(len(X)):
    x = X[i,1]
    t_0 = X[i,0]
    lower = t_0 - 1
    uper = 1 - t_0    
    sol[i] = np.where((x<lower) | (x>uper),0,1) * 1/(2-2*t_0) 
  return sol.reshape((len(X),1))

def convolution(t):
  x_evaluate = np.array([(t[0],tt) for tt in np.linspace(-1.5,1.5,Nx)])          
  u_eval = solution(x_evaluate).reshape(-1)        
  gauss = gaussian(x_evaluate[:,1],0.02)  
  conv = np.convolve(u_eval,gauss,mode='same')  
  sol = ((conv/np.max(conv))*np.max(u_eval)).reshape((len(u_eval),1))
  t = np.delete(t,0)
  for t_0 in t:            
    x_evaluate = np.array([(t_0,tt) for tt in np.linspace(-1.5,1.5,Nx)])    
    u_eval = solution(x_evaluate).reshape(-1)        
    gauss = gaussian(x_evaluate[:,1],0.02)        
    conv = np.convolve(u_eval,gauss,mode='same')  
    sol = np.append(sol,((conv/np.max(conv))*np.max(u_eval)).reshape((len(u_eval),1)),axis=0)    
  return sol

Lx = 2
Nx = 200
Nt = 100 #Hay un error cuando Nx > Nt para armar los flags

t = np.linspace(0,0.8,Nt)
x = np.linspace(-1,1,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)

Y = convolution(t) #[u(t_0,x_0),u(t_1,x_1),...]

lambda_data = np.zeros(Nt*Nx) #[1,0,0,..]
lambda_data[:Nx] = 1

lambda_phys = np.ones(Nt*Nx)
lambda_phys[:Nx] = 0 #[0,1,1,..]

bc = np.zeros(Nt*Nx)
bc[0] = 1
bc[-1] = 1
lambda_bc = np.tile(bc,Nt) #[1,0,..Nx..0,1] Nt veces 

flags = np.repeat(np.arange(Nt),Nx)

sigma = 0.02
alpha   = 0.0
tot_eps = 5000
lam = sigma/2
eq_params = [Lx/Nx]
eq_params = [np.float32(p) for p in eq_params] #eq_params es el diferencial de x que paso para calcular la integral

PINN.train(X, Y, opinion_model,
           epochs=tot_eps,
           eq_params=eq_params,
           batch_size=Nx,
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,
            lambda_bc=lambda_bc,
           flags=flags,               # Separa el dataset a cada t
           rnd_order_training=False,  # No arma batches al hacer
           alpha=alpha,
           verbose=True,
           timer=False)



#fields = (PINN.model(X)[0]).numpy()



