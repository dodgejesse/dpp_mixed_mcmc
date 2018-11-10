import tensorflow as tf
import numpy as np




def find_dpp_max(k, d, sigma, logging):
  tf.reset_default_graph()
  ks = tf.Variable(tf.random_uniform([k,d]))
  ks_init = ks.initializer


  with tf.Session() as sess:
    sess.run(ks_init)
    initial_points = ks.eval()
    sq_dists = (-0.5 * tf.reduce_sum(tf.pow(tf.reshape(ks,[k,1,d]) - tf.reshape(ks,[1,k,d]),2),axis=2)) / sigma ** 2
    my_kernel = tf.exp(sq_dists)
    cur_det = -tf.linalg.det(my_kernel)
    
    
    logging['start_det'].append(cur_det.eval())
    starting_det = cur_det.eval()
    try:
      logging['avg_grad'].append(tf.reduce_mean(tf.abs(tf.gradients(cur_det, [ks])[0])).eval())
    except:
      logging['avg_grad'].append(False)

    
    # for using L-BFGS-B (bounded L-BFGS to a box)
    box_bounds = {ks:(0,1)}
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(
      cur_det, var_to_bounds=box_bounds, method='L-BFGS-B')

    #try:
    optimizer.minimize()
    #logging['minimized'].append(True)
    #except:
    #logging['minimized'].append(False)
    
    logging['end_det'].append(cur_det.eval())
    logging['det_diff'].append(-cur_det.eval() + starting_det)
    logging['end_ks'].append(ks.eval())
    logging['ks_diffs'].append(ks.eval() - initial_points)
    

def init_logging():
  logging = {}
  things_to_log = ['start_det', 'avg_grad', 'minimized', 'end_det', 'det_diff', 'end_ks', 'ks_diffs']
  for thing in things_to_log:
    logging[thing] = []
  return logging


k = 10
d = 3
sigma = 1
logging = init_logging()
tf.set_random_seed(1234)
for i in range(5):
  #import pdb; pdb.set_trace()
  find_dpp_max(k,d,sigma,logging)
print logging['det_diff']
for item in logging['ks_diffs']:
  print(item)
