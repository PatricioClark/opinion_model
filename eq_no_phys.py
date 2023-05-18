import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np

@tf.function
def test(model, coords, params):
    """ Opinion model
    Assumes coords is ordered in space and at constant t
    """

    # Calculate F(t_0, x)
    #dx = params[0]
    #Yp = model(coords)[0]            
    #fx = tf.cumsum(Yp)        
    #fx = fx*dx #Saque el tf.constant(fx) para solucionar el error TypeError: Expected any non-tensor type, got a tensor instead
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        Yp = model(coords)[0]
        u   = Yp[:,0] 
        
    del tape

    f = u
    return [f]
