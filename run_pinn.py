"PINN implementation of opinion model"
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import opinion_model 
import numpy as np
import time as time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks



#lr = keras.optimizers.schedules.ExponentialDecay(1e-4, 1000, 0.9)
lr = 1e-4
layers  = [2] + 3*[64] + [2]

PINN = PhysicsInformedNN(layers,
                         dest='./',
                         activation='tanh',
                         optimizer=keras.optimizers.Adam(lr),
                         #optimizer='lbfgs',
                         restore=True)
PINN.model.summary()

def cte_validation(self,X,u):    
    # Definimos una función que el código después llama
    # El único parametro que le pasa código es el número de epoch
    # El resto lo definimos al generar la función
    def validation(ep):        
        # Get prediction                                
        Y  = self.model(X)[0].numpy()

        u_p = Y[:,0]       

        # True value
        sol  = u(X)
        
        # Error global
        err  = np.sqrt(np.mean((u_p-sol)**2))/np.std(sol)
        
        # Loss functions
        output_file = open(self.dest + 'validation.dat', 'a')
        print(ep,err,file=output_file)
        output_file.close()       
        
        dx = 4/len(X)
        fx = tf.cumsum(u_p)
        fx = fx*dx     
        Ix = tf.ones_like(u_p)*(fx[-1])           

        output_file_1 = open(self.dest + 'mass_conservation.dat', 'a')
        print(ep,f'{Ix[-1].numpy()}',file=output_file_1)
        output_file_1.close()    
        
    return validation
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
def convolution(X):
  t = len(np.unique(X[:,0]))  
  Nx = int(len(X)/t)  
  sol = np.zeros((Nx*t,1))
  for i in range(t):
    x_eval = X[i*Nx:(i+1)*Nx]          
    u_eval = solution(x_eval).reshape(-1)        
    gauss = gaussian(x_eval[:,1],0.05)  
    conv = np.convolve(u_eval,gauss,mode='same')  
    sol[i*Nx:(i+1)*Nx] = ((conv/np.max(conv))*np.max(u_eval)).reshape((len(u_eval),1))            
  return sol
def linear(x):
   sol = np.where((x<-1) | (x>1),0,1) * ((x+1)/2)
   return sol.reshape(len(x),1)
def dif_fin(x):   
   dx = (x[:,1][1] - x[:,1][0])
   u = convolution(x).reshape(len(x[:,1]))
   u[1] = u[0]
   u[-2] = u[-1]
   f = np.cumsum(u)*dx
   F = u*(2*f - 1)      
   return np.gradient(F, dx).reshape(len(x[:,1]),1)
def collocation_distribution(x,s,peaks):    
    g = 0
    for i in range(len(peaks)):
      g += 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -(x - peaks[i])**2 / ( 2. * s**2 ) )            
    return g
def domain(t,x,N):      
   t_0 = t[0]
   x_0 = np.array([(t_0,tt) for tt in x])
   u_t = np.abs(dif_fin(x_0))
   peaks_pos = find_peaks(u_t.reshape(len(u_t)),prominence=1.5)
   peak_location = [x[i] for i in peaks_pos[0]]   
   dist = collocation_distribution(x_0[:,1],0.2,peak_location)   
   uniform_samples = np.random.uniform(0,1,N)
   cdf = np.cumsum(dist)   
   collocation_points = np.interp(uniform_samples, cdf / cdf[-1], x)      
   x1 = np.linspace(x[0],x[-1],100 - N)
   domain = np.concatenate([x1,collocation_points])   
   return domain
   

Lx = 4 
Nx = 100
Nt = 500

t = np.linspace(0,0.05,Nt)
x = np.linspace(-2,2,Nx)

space = np.sort(domain(t,x,40))

T,X = np.meshgrid(t,space)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)

Y = np.hstack((convolution(X), convolution(X))) #[u(t_0,x_0),u(t_1,x_1),...]

lambda_data = np.zeros(Nt*Nx) #[1,0,0,..]
lambda_data[:Nx] = 1e5

lambda_phys = np.ones(Nt*Nx)
lambda_phys[:Nx] = 0 #[0,1,1,..]

bc = np.zeros(Nx)
bc[:5] = 1
bc[-5:] = 1
lambda_bc = np.tile(bc,Nt)

n_t_in_batch = Nt #Nt tiene que ser divisble por batches
flags = np.repeat(np.arange(Nt/n_t_in_batch),Nx*n_t_in_batch)

alpha = 0.0
tot_eps = 100000
eq_params = [Lx/Nx,n_t_in_batch]
#eq_params = [np.float32(p) for p in eq_params] 

PINN.validation = cte_validation(PINN,X,convolution)

t1 = time.time()
PINN.train(X, Y, opinion_model,
           epochs=tot_eps,
           batch_size=n_t_in_batch*Nx,
           eq_params=eq_params,           
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,
           lambda_bc=lambda_bc, 
           flags=flags,               # Separa el dataset a cada t
           rnd_order_training=False,  # No arma batches al hacer
           alpha=alpha,
           verbose=False,            
           valid_freq=5000,
           timer=False,
           data_mask=[True,False])

t2 = time.time()
print(int(t2-t1)/60)
