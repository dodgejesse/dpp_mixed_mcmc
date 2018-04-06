import numpy as np
import sklearn.metrics
import tensorflow as tf



vector = tf.Variable([7., 7.], 'vector')

# Make vector norm as small as possible.
loss = tf.reduce_sum(tf.square(vector))
# Ensure the vector's y component is = 1.
equalities = [vector[1] - 1.]
# Ensure the vector's x component is >= 1.
inequalities = [vector, 1. - vector]

box_bounds = {vector:(0,1)}

# Our default SciPy optimization algorithm, L-BFGS-B, does not support
# general constraints. Thus we use SLSQP instead.
optimizer = tf.contrib.opt.ScipyOptimizerInterface(
    loss, var_to_bounds=box_bounds, method='L-BFGS-B')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    optimizer.minimize()
    print loss.eval()
    print vector.eval()

# The value of vector should now be [1., 1.].




#ks = np.array([[1,1],[.5,.5],[0,0]])

#print ks

#L_Y = sklearn.metrics.pairwise.rbf_kernel(ks,ks,sigma=25)
#print L_Y

