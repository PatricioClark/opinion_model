""" PINN implementation of opinion model """

import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

from pinn      import PhysicsInformedNN
from equations import opinion_model
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio


#lr = keras.optimizers.schedules.ExponentialDecay(1e-4, 1000, 0.9)
lr = 1e-7
layers  = [2] + 3*[64] + [1]

PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='tanh',
                         optimizer=keras.optimizers.Adam(lr),
                         restore=True)
def gif_sol(t):  
  filenames = []
  for i in t:    
    dom = np.array([(i,tt) for tt in np.linspace(np.min(x),np.max(x),Nx)])
    solution = convolution(dom)
    pinn = PINN.model(dom)[0]
    plt.title(f'Solucion a t = {i}')
    plt.plot(dom[:,1],pinn,label = 'PINN')
    plt.plot(dom[:,1],solution,label = 'Solucion Real')
    plt.legend()
    # create file name and append it to a list
    filename = f'{i}.png'    
    for j in range(10):       
       filenames.append(filename)                 
    # save frame
    plt.savefig(filename)
    plt.close()  
  # build gif
  with imageio.get_writer('solution.gif', mode='I',duration=0.001) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)          
  # Remove files
  for filename in set(filenames):    
    os.remove(filename)
def gif_val(t):  
  filenames = []
  for i in t:    
    dom = np.array([(i,tt) for tt in np.linspace(np.min(x),np.max(x),Nx)])
    solution = convolution(dom)
    pinn = PINN.model(dom)[0]
    validation = ((pinn-solution)**2)/np.std(solution)
    plt.title(f'Solucion a t = {i}')    
    plt.plot(dom[:,1],validation,label = 'Error')
    plt.legend()
    # create file name and append it to a list
    filename = f'{i}.png'    
    for j in range(10):       
       filenames.append(filename)                 
    # save frame
    plt.savefig(filename)
    plt.close()  
  # build gif
  with imageio.get_writer('validation.gif', mode='I',duration=0.0000001) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)          
  # Remove files
  for filename in set(filenames):    
    os.remove(filename)    
def gaussian(x , s):
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
  s = 0.1
  t = len(np.unique(X[:,0]))  
  Nx = int(len(X)/t)  
  sol = np.zeros((Nx*t,1))
  for i in range(t):
    x_eval = X[i*Nx:(i+1)*Nx]          
    u_eval = solution(x_eval).reshape(-1)        
    gauss = gaussian(x_eval[:,1],s)  
    conv = np.convolve(u_eval,gauss,mode='same')  
    sol[i*Nx:(i+1)*Nx] = ((conv/np.max(conv))*np.max(u_eval)).reshape((len(u_eval),1))            
  return sol
def Euler(condicion_inicial,tiempo_final,n_pasos_temporales,espacio):
    h = tiempo_final/n_pasos_temporales
    tiempo = 0
    solucion = condicion_inicial
    for j in range(n_pasos_temporales):
        tiempo += h
        x = np.array([(tiempo,tt) for tt in espacio])
        F = convolution(x)
        solucion += h*F
    return solucion

Lx = 2 
Nx = 100
Nt = 20

t = np.linspace(0,0.01,Nt)
x = np.linspace(-2,2,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)

#Solucion real y solucion de la rex
Y = convolution(X) 
fields = PINN.model(X)[0]

#Graficos
sol_3D_show = False
model_3D_show = False
model_show = False # Solucion de la red en todo el espacio 
sol_show = False # Solucion real en todo el espacio
loss_val = True # Funcion de perdida + Validation
val = False # Validation sola
loss = True # loss sola
cond_in = True # Condicion inicial
x_0 = True # Miro la solucion a un tiempo t_0
error_loc = True
loss_eq = False
Euler_graph = False
#gif_sol(t)
#gif_val(t)

#Veo la solucion a tiempo inicial.
t_ini = 0.0
x_eval_1 = np.array([(t_ini,tt) for tt in x])
u_eval_1 = convolution(x_eval_1)
fields_eval_1 = PINN.model(x_eval_1)[0]

##Veo la solucion a tiempo final.
t_fijo = t[-1]
x_eval_2 = np.array([(t_fijo,tt) for tt in x])
u_eval_2 = convolution(x_eval_2)
fields_eval_2 = PINN.model(x_eval_2)[0]

#Validacion en el tiempo final
val_local = ((u_eval_2-fields_eval_2)**2)/np.std(fields_eval_2)


dx = 4/Nx
#Integral a tiempo inicial.
fx = tf.cumsum(fields_eval_1)
fx_true = tf.cumsum(u_eval_1)
fx = fx*dx
fx_true = fx_true*dx 

I_pinn = fx[-1]
I_true = fx_true[-1]

#Integral a tiempo final.
fx_2 = tf.cumsum(fields_eval_2)
fx_true_2 = tf.cumsum(u_eval_2)
fx_2 = fx_2*dx
fx_true_2 = fx_true_2*dx 

I_pinn_2 = fx_2[-1]
I_true_2 = fx_true_2[-1]

#Derivadas parciales de la solucion de la red.
coords = tf.convert_to_tensor(x_eval_1)

with tf.GradientTape(persistent=True) as tape1:
      tape1.watch(coords)           
      Yp = PINN.model(coords)[0]
      u_p   = Yp[:,0]              
      uf  = u_p*(2*fx - 1)                                                        
grad_u  = tape1.gradient(u_p, coords,unconnected_gradients=tf.UnconnectedGradients.ZERO)                         
u_t = grad_u[:,0]
u_x = grad_u[:,1]                         
grad_uf = tape1.gradient(uf,coords,unconnected_gradients=tf.UnconnectedGradients.ZERO)           
uf_x = grad_uf[:,1]           
           
del tape1
        
f1 = uf_x
f2 = u_t 

#Evaluo el metodo de euler
euler = Euler(u_eval_1,t_fijo,Nt,x)

if sol_3D_show:
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0].flatten(), X[:,1].flatten(), convolution(X).flatten())
    ax.view_init(10,90)
if model_3D_show:  
    fig = plt.figure(figsize = (10,10))
    plt.title('Model')
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0].flatten(), X[:,1].flatten(), fields)
    ax.view_init(10,90)
if sol_show:
    solution = np.reshape(Y,(Nt,Nx))    
    for _ in range(3):
      solution = np.rot90(solution)
    plt.figure(figsize=(5,5))
    plt.imshow(solution, cmap = 'hot',extent=[0,np.max(t),t[0],t[-1]],aspect=0.4)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
if model_show:   
    model = np.reshape(fields,(Nt,Nx))    
    for _ in range(3):
      model = np.rot90(model)
    plt.figure(figsize=(5,5))
    plt.imshow(model, cmap = 'hot',extent=[0,np.max(t),t[0],t[-1]],aspect=0.4)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
if loss_val:
  fig, ax1 = plt.subplots()
  
  ax1.set_xlabel('Epochs')  
  ax1.set_ylabel('Loss', color='red')
  out = np.loadtxt('output.dat', unpack=True)
  lns1 = ax1.semilogy(out[0], out[2],color='red',label='Loss function')
  
  ax2 = ax1.twinx()  

  ax2.set_ylabel('Validation', color='blue')
  ax2.tick_params(axis='y', labelcolor='blue')  
  out_1 = np.loadtxt('validation.dat', unpack=True)
  lns2 = ax2.semilogy(out_1[0], out_1[1], color='blue',label='Validation')
  
  lns = lns1+lns2
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc=0)        
if loss:
  plt.figure()
  out = np.loadtxt('output.dat', unpack=True)
  plt.semilogy(out[0], out[1],label='Loss data')
  plt.semilogy(out[0], out[2],label='Loss phys')
  plt.semilogy(out[0], out[3],label='Loss bc')
  plt.title('Training loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
if val:
    plt.figure()  
    out_1 = np.loadtxt('validation.dat', unpack=True)
    plt.semilogy(out_1[0], out_1[1], label='Validation')          
    plt.title('Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')    
if cond_in:
  plt.figure()
  plt.title('Condicion Inicial')
  plt.plot(x_eval_1[:,1],fields_eval_1,label = 'PINN')
  plt.plot(x_eval_1[:,1],u_eval_1,label = 'Solucion Real')
  plt.xlabel('X')
  plt.ylabel('u(X)')
  plt.grid()
  plt.legend()
if x_0:
  plt.figure()
  plt.title(f'Solucion a t = {t_fijo}')  
  plt.plot(x_eval_2[:,1],fields_eval_2, label = 'PINN')
  plt.plot(x_eval_2[:,1],u_eval_2, label = 'Solucion Real')    
  plt.xlabel('X')
  plt.ylabel('$u(x,t = t_{0})$')
  plt.grid()
  plt.legend()
if error_loc:
   plt.figure()
   plt.title(f'Validacion a t = {t_fijo}')  
   plt.plot(x_eval_2[:,1],val_local, label = 'Error')   
   plt.xlabel('X')
   plt.ylabel('Error')
   plt.grid()
   plt.legend()
if loss_eq:
  plt.figure()
  out = np.loadtxt('mass_conservation.dat', unpack=True)    
  plt.semilogy(out[0], out[1],label='Mass conservation')
  plt.title('Terms')
  plt.xlabel('Epochs')
  plt.ylabel('Mass')
  plt.legend()
if Euler_graph:  
  plt.figure()
  plt.title(f'Euler')  
  plt.plot(x_eval_2[:,1],fields_eval_2, label = 'PINN')
  plt.plot(x_eval_2[:,1],u_eval_2, label = 'Solucion Real')    
  plt.plot(x_eval_2[:,1],euler,'o',label = 'euler')
  plt.xlabel('X')
  plt.ylabel('$u(x,t = t_{0})$')
  plt.legend()
   

plt.figure()
plt.title('Residuales a tiempo final')
plt.plot(x_eval_2[:,1],u_t.numpy() - u_x.numpy().reshape(100)*(2*fx_2.numpy().reshape(100) - 1),label = '$u_{t} - uf_{x}$')
plt.plot(x_eval_2[:,1],u_x.numpy().reshape(100)*(2*fx_2.numpy().reshape(100) - 1),label = '$uf_{x}$')
plt.plot(x_eval_2[:,1],u_t.numpy(),label = '$u_{t}$')
plt.legend()
plt.grid()


plt.show()