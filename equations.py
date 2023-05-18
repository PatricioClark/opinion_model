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
    lam = params[1]
    Yp = model(coords)[0]            
    fx = tf.cumsum(Yp)        
    fx = fx*dx 
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(coords)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(coords)
            Yp = model(coords)[0]
            u   = Yp[:,0] 
            uf  = u*(2*fx - 1)

        #LHS
        grad_u  = tape1.gradient(u, coords)
        u_t = grad_u[:,0]
        u_x = grad_u[:,1]

        #RHS
        grad_uf = tape1.gradient(uf, coords)
        uf_x = grad_uf[:,1]

        del tape1

    u_xx = tape2.gradient(u_x, coords)[:,1]
    
    del tape2

    f = u_t - uf_x - (lam) * u_xx
    return [f]
