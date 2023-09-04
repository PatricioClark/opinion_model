import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np

@tf.function
def opinion_model(model, coords, params):
    """ Opinion model
    Assumes coords is ordered in space and at constant t
    """
    # Calculate F(t_0, x)    
    dx = params[0]             
    Yp = model(coords)[0]      
    Fp = Yp[:,1]           
    split = tf.split(Yp[:,0],params[1],axis=0)
    s1 = tf.cumsum(split,axis=1)
    s2 = s1 * dx
    l = [s2[i] for i in range(len(s2))]
    Fx =  tf.concat(l,axis = 0)
    Ix_batch = [s2[i][-1] for i in range(len(s2))]    
    Ix_batch_t = [tf.ones_like(s2[0])*Ix_batch[i] for i in range(len(Ix_batch))]    
    Ix = tf.concat(Ix_batch_t,axis = 0)
        
    with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u   = Yp[:,0] 
            F   = Yp[:,1] 
            uf  = u*(2*F - 1)

    #LHS
    grad_u  = tape1.gradient(u, coords)
    u_t = grad_u[:,0]
        
    #RHS
    grad_uf = tape1.gradient(uf, coords)
    uf_x = grad_uf[:,1]

    del tape1    

    f = u_t - uf_x     
    eq_f = Fx - Fp
           
    return [f,eq_f,tf.reshape(Ix,f.shape) - 1]

