from __future__ import print_function
from __future__ import division

import os
import sys
import time

import numpy as np
import tensorflow as tf

import roadway.input as input

NO_CLASSES = 841
checkpoint_dir = 'checkpoints'
BATCH_SIZE=40
INITIAL_LR_SOFTMAX = 0.001
INITIAL_LR_FC = 0.0001
INITIAL_LR_CONV = 0.000001
LR_DECAY_FACTOR = 0.0000001


steps_per_epoch = 2060 # int (82660/ 40)
# Factor of 3 since we have a separate minimize for softmax, FC and conv layers
# Learning rate would be decayed after 5 epochs
decay_steps = steps_per_epoch * 5 * 3

def inference(images):
    
  # conv1_1
  with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                            stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], 
                        dtype=tf.float32), name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)
  # conv1_2
  with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                              stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), 
                                    name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)
    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool1')
  # conv2_1
  with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                              stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128],
                        dtype=tf.float32), name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)


  # conv2_2
  with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128],
                                              dtype=tf.float32, stddev=1e-1),
                                              name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                        name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)


    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool2')
  # conv3_1
  with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], 
                                            dtype=tf.float32,
                                            stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)

  # conv3_2
  with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                          trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)
        
# conv3_3
  with tf.name_scope('conv3_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], 
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_3 = tf.nn.relu(out, name=scope)

  # pool3
  pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool3')
  
  # conv4_1
  with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                    name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)
  
  # conv4_2
  with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)

  # conv4_3
  with tf.name_scope('conv4_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                    name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_3 = tf.nn.relu(out, name=scope)
    
  # pool4
  pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool4')

  # conv5_1
  with tf.name_scope('conv5_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_1 = tf.nn.relu(out, name=scope)

  # conv5_2
  with tf.name_scope('conv5_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512],
                                            dtype=tf.float32,
                                            stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_2 = tf.nn.relu(out, name=scope)

  # conv5_3
  with tf.name_scope('conv5_3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], 
                                            dtype=tf.float32, stddev=1e-1),
                                            name='weights')
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                     name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv5_3 = tf.nn.relu(out, name=scope)

  # pool5
  pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                  padding='SAME', name='pool4')
        
  # fc1
  with tf.name_scope('fc1') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
    fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                   name='biases')
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    fc1 = tf.nn.relu(fc1l)
  
  # fc2
  with tf.name_scope('fc2') as scope:
    fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                            dtype=tf.float32,
                                            stddev=1e-1), name='weights')
    fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                   name='biases')
    fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
    fc2 = tf.nn.relu(fc2l)
    
  # fc3
  with tf.name_scope('fc3') as scope:
    fc3w = tf.Variable(tf.truncated_normal([4096, NO_CLASSES],
                                            dtype=tf.float32,
                                            stddev=1e-1), name='weights')
    fc3b = tf.Variable(tf.constant(1.0, shape=[NO_CLASSES], 
                                        dtype=tf.float32), name='biases')
            
    # fc3l -> softmadx_linear
    fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
  
  return fc3l

def loss_function(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(loss, global_step):

  trainable_vars = tf.trainable_variables()
  
  convs = trainable_vars[:26]
  fcs = trainable_vars[26:30]
  softmax = trainable_vars[30:]
  lr_softmax = (
    tf.train.exponential_decay(INITIAL_LR_SOFTMAX, global_step, decay_steps,
                              LR_DECAY_FACTOR, staircase=True)
  )
  lr_fc = (
    tf.train.exponential_decay(INITIAL_LR_FC, global_step, decay_steps,
                              LR_DECAY_FACTOR, staircase=True)
  )
  lr_conv = (
    tf.train.exponential_decay(INITIAL_LR_CONV, global_step, decay_steps,
                              LR_DECAY_FACTOR, staircase=True)
  )

  op1 = tf.train.GradientDescentOptimizer(lr_softmax)
  op2 = tf.train.GradientDescentOptimizer(lr_fc)
  op3 = tf.train.GradientDescentOptimizer(lr_conv)

  grads = tf.gradients(loss, softmax + fcs + convs[24:])
  softmax_grads = grads[:(len(softmax))]
  fcs_grads = grads[len(softmax):len(softmax)+len(fcs)]
  convs_grads  = grads[len(softmax)+len(fcs):]

  train_op1 = op1.apply_gradients(zip(softmax_grads, softmax),
                                  global_step=global_step)
  train_op2 = op2.apply_gradients(zip(fcs_grads, fcs),
                                  global_step=global_step)
  train_op3 = op3.apply_gradients(zip(convs_grads, convs[24:]),
                                  global_step=global_step)

  train_op = tf.group(train_op1, train_op2, train_op3)
  return train_op

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

def run_training():
  
  with tf.Graph().as_default():
    with tf.device('/gpu:0'):
      global_step = tf.Variable(0, trainable=False)
      images, labels = input.inputs(True, BATCH_SIZE, 20)
      logits = inference(images)
      loss = loss_function(logits, labels)
      train_op = train(loss, global_step)

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
      #sess.run(global_step.assign(step))
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
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time
        step += 1
        if step % 20 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step,
                                                     loss_value,
                                                     duration))
        # 1 - Epoch
        if step % 2060 == 0:
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
