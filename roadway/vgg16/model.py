from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer,
			  dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized variable with weight decay
  """
  var = _variable_on_cpu(
        name, shape, 
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _conv_layer(input, shape, strides, scope):
  """Helper to setup a convolution layer (CONV + RELU)
  """
  #kernel = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32,
  #                                        stddev=1e-1), name='weights')
  kernel = _variable_with_weight_decay('weights',
                                        shape=shape,
                                        stddev=1e-1,
                                        wd=0.0)
  conv = tf.nn.conv2d(input, kernel, strides, padding='SAME')
  biases = _variable_on_cpu('biases', shape=[shape[3]], 
                            initializer=tf.constant_initializer(0.0))
  out = tf.nn.bias_add(conv, biases)
  return tf.nn.relu(out, scope.name)

def inference(images, no_classes, keep_prob=1.0):
    
  # conv1_1
  with tf.variable_scope('conv1_1') as scope:
    conv1_1 = _conv_layer(images, [3,3,3,64], [1,1,1,1], scope)
  # conv1_2
  with tf.variable_scope('conv1_2') as scope:
    conv1_2 = _conv_layer(conv1_1, [3,3,64,64], [1,1,1,1], scope)
    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool1')
  # conv2_1
  with tf.variable_scope('conv2_1') as scope:
    conv2_1 = _conv_layer(pool1, [3, 3, 64, 128], [1, 1, 1, 1], scope)

  # conv2_2
  with tf.variable_scope('conv2_2') as scope:
    conv2_2 = _conv_layer(conv2_1, [3, 3, 128, 128], [1, 1, 1, 1], scope)
    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool2')
  # conv3_1
  with tf.variable_scope('conv3_1') as scope:
    conv3_1 = _conv_layer(pool2, [3, 3, 128, 256], [1, 1, 1, 1], scope)

  # conv3_2
  with tf.variable_scope('conv3_2') as scope:
    conv3_2 = _conv_layer(conv3_1, [3, 3, 256, 256], [1, 1, 1, 1], scope)
        
  # conv3_3
  with tf.variable_scope('conv3_3') as scope:
    conv3_3 = _conv_layer(conv3_2, [3, 3, 256, 256], [1, 1, 1, 1], scope)
   
  # pool3
  pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool3')
  
  # conv4_1
  with tf.variable_scope('conv4_1') as scope:
    conv4_1 = _conv_layer(pool3, [3, 3, 256, 512], [1, 1, 1, 1], scope)

  # conv4_2
  with tf.variable_scope('conv4_2') as scope:
    conv4_2 = _conv_layer(conv4_1, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # conv4_3
  with tf.variable_scope('conv4_3') as scope:
    conv4_3 = _conv_layer(conv4_2, [3, 3, 512, 512], [1, 1, 1, 1], scope)
    
  # pool4
  pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool4')

  # conv5_1
  with tf.variable_scope('conv5_1') as scope:
    conv5_1 = _conv_layer(pool4, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # conv5_2
  with tf.variable_scope('conv5_2') as scope:
    conv5_2 = _conv_layer(conv5_1, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # conv5_3
  with tf.variable_scope('conv5_3') as scope:
    conv5_3 = _conv_layer(conv5_2, [3, 3, 512, 512], [1, 1, 1, 1], scope)

  # pool5
  pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool4')
        
  # fc1
  keep_prob = tf.constant(keep_prob)
  with tf.variable_scope('fc1') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    fc1w = _variable_with_weight_decay('weights', shape=[shape, 4096],
                                        stddev=1e-1, wd=0.0)
    fc1b = _variable_on_cpu('biases', [4096], tf.constant_initializer(1.0))
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    fc1 = tf.nn.relu(fc1l)
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

  
  # fc2
  with tf.variable_scope('fc2') as scope:
    fc2w = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                        stddev=1e-1, wd=0.0) 
    fc2b = _variable_on_cpu('biases', [4096], tf.constant_initializer(1.0))
    fc2l = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
    fc2 = tf.nn.relu(fc2l)
    fc2_drop = tf.nn.dropout(fc2, keep_prob)

  # fc3
  with tf.variable_scope('fc3') as scope:
    fc3w = _variable_with_weight_decay('weights', shape=[4096, no_classes],
                                        stddev=1e-1, wd=0.0)
    fc3b = _variable_on_cpu('biases', [no_classes],
                            tf.constant_initializer(1.0))
    # fc3l -> softmax_linear
    fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
  
  return fc3l

def loss_function(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  total_loss =  tf.add_n(tf.get_collection('losses'), name='total_loss')
  with tf.control_dependencies([total_loss]):
    total_loss = tf.identity(total_loss)
  return total_loss

def load_weights(weight_file, sess):
  parameters = tf.trainable_variables()
  weights = np.load(weight_file)
  keys = sorted(weights.keys())
  for i, k in enumerate(keys):
    # Skipping last layer
    if k == 'fc8_W' or k == 'fc8_b':
      continue
    #print (i, k, np.shape(weights[k]))
    sess.run(parameters[i].assign(weights[k]))
