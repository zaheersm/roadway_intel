from __future__ import print_function
from __future__ import division

import os
import sys
import time

import numpy as np
import tensorflow as tf

import input

NO_CLASSES = 841
checkpoint_dir = 'checkpoints'
BATCH_SIZE = 80
INITIAL_LR_SOFTMAX = 0.0001
INITIAL_LR_FC = 0.0001
INITIAL_LR_CONV = 0.0001
LR_DECAY_FACTOR = 0.1


steps_per_epoch = 1030 # int (82660/ 80)
# Factor of 3 since we have a separate minimize for softmax, FC and conv layers
# Learning rate would be decayed after 3 epochs
decay_epochs = 30
decay_steps = steps_per_epoch * decay_epochs * 3

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized variable with weight decay
  """
  var = _variable_on_cpu(
        name, shape, 
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
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

def inference(images, keep_prob=1.0):
    
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
    fc3w = _variable_with_weight_decay('weights', shape=[4096, NO_CLASSES],
                                        stddev=1e-1, wd=0.0)
    fc3b = _variable_on_cpu('biases', [NO_CLASSES], tf.constant_initializer(1.0))            
    # fc3l -> softmax_linear
    fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
  
  return fc3l

def loss_function(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits, labels, name='cross_entropy_per_example')
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
    print (i, k, np.shape(weights[k]))
    sess.run(parameters[i].assign(weights[k]))

def _get_learning_rate(INITIAL_LR, global_step):
  """ Helper to get learning_rate
  """
  return tf.train.exponential_decay(INITIAL_LR, global_step, decay_steps,
                            LR_DECAY_FACTOR, staircase=True)

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def run_training():
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False)
    lr_softmax = _get_learning_rate(INITIAL_LR_SOFTMAX, global_step)
    lr_fc = _get_learning_rate(INITIAL_LR_FC, global_step)
    lr_conv = _get_learning_rate(INITIAL_LR_CONV, global_step)
    op1 = tf.train.GradientDescentOptimizer(lr_softmax)
    op2 = tf.train.GradientDescentOptimizer(lr_fc)
    op3 = tf.train.GradientDescentOptimizer(lr_conv)
  
    tower_softmax_grads = []
    tower_fcs_grads = []
    tower_convs_grads = []

    images, labels = input.inputs(True, BATCH_SIZE, 9)
    split_images = tf.split(0, 2, images)
    split_labels = tf.split(0, 2, labels)
    loss = []
    for i in xrange(2):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('tower',i)) as scope:
          images, labels = input.inputs(True, BATCH_SIZE, 20)
          logits = inference(split_images[i], 0.5)
          loss.append(loss_function(logits, split_labels[i]))

          tf.get_variable_scope().reuse_variables()

          trainable_vars = tf.trainable_variables()
          convs = trainable_vars[24:26]
          fcs = trainable_vars[26:30]
          softmax = trainable_vars[30:]
          softmax_grads = op1.compute_gradients(loss[i], softmax)
          tower_softmax_grads.append(softmax_grads)
          fcs_grads = op2.compute_gradients(loss[i], fcs)
          tower_fcs_grads.append(fcs_grads)
          convs_grads = op3.compute_gradients(loss[i], convs)
          tower_convs_grads.append(convs_grads)

    softmax_grads = average_gradients(tower_softmax_grads) 
    fcs_grads = average_gradients(tower_fcs_grads)
    convs_grads = average_gradients(tower_convs_grads)
    
    train_op1 = op1.apply_gradients(softmax_grads, global_step=global_step)
    train_op2 = op2.apply_gradients(fcs_grads, global_step=global_step)
    train_op3 = op3.apply_gradients(convs_grads, global_step=global_step)

    train_op = tf.group(train_op1, train_op2, train_op3)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_local_variables()))

    saver = tf.train.Saver(tf.trainable_variables())
    step = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print ('Restoring from checkpoint %s' % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      sess.run(global_step.assign(step*3))
    else:
      print ('No checkpoint file found\nRestoring ImageNet '+\
             'Pretrained: vgg16_weights.npz')
      load_weights('vgg16_weights.npz', sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print ('Training begins..')
    try:
      while not coord.should_stop():
        start_time = time.time()
        _, loss_value, global_step_val = sess.run([train_op,
                                                  loss[0],
                                                  global_step])
        duration = time.time() - start_time
        step += 1
        if step % 20 == 0:
          print('Step %d(%d): loss = %.2f (%.3f sec)' % (step,
                                                     global_step_val,
                                                     loss_value,
                                                     duration))
        # 1 - Epoch
        if step % 1030 == 0:
          print ('Saving Model')
          checkpoint_path = os.path.join('checkpoints', 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
      print ('tf.errors.OutOfRangeError')
    finally:
      print('Done training for %d steps.' % (step))
      # When done, ask the threads to stop.
      coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
  run_training()
