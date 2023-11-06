import tensorflow as tf
tf.keras.backend.set_floatx('float32')
#import numpy as np        
            
@tf.function        
def tf_reduceat(data, at_array, axis=0):
    split_data_1 = tf.split(data/2, at_array, axis=axis)
    split_data = tf.concat([split_data_1[i] + split_data_1[i-1] for i in range(1,len(split_data_1))],axis = 0)    
    return tf.concat([tf.expand_dims(split_data[0],axis=0), split_data],axis = 0) 


@tf.function
def opinion_model(model, coords, params):
    """ Opinion model
    Assumes coords is ordered in space and at constant t
    """
    # Calculate F(t_0, x)        
    #dx = params[0]             
    Yp = model(coords)[0]      
    Fp = Yp[:,1]           
    X_split = tf.split(coords[:,1],params[1],axis=0)
    X_diff = tf.experimental.numpy.diff(X_split)
    X_split_diff = [tf.concat([tf.expand_dims(X_diff[i][0],axis=0), X_diff[i]],axis = 0) for i in range(len(X_diff))]

    Y_split = tf.split(Yp[:,0],params[1],axis=0)
    trapezoid = [tf_reduceat(Y_split[i],len(Y_split[i])) for i in range(len(Y_split))]

    s1 = tf.multiply(trapezoid,X_split_diff)    
    s2 = tf.cumsum(s1,axis=1)
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

