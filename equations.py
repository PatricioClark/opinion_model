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
    fx = tf.cumsum(Yp)
    fx = tf.constant(fx)*dx

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        Yp = model(coords)[0]
        u   = Yp[:,0] 
        uf  = u*(2*fx - 1)

    # LHS
    grad_u  = tape.gradient(u, coords)
    u_t = grad_u[:,0]

    # RHS
    grad_uf = tape.gradient(uf, coords)
    uf_x = grad_uf[:,1]

    del tape

    f = u_t - uf_x
    return [f]
