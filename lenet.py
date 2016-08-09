"""
LeNet architecture

Conv - Pool - ReLU - Conv - Pool - ReLU - FC - FC Softmax
"""
from __future__ import print_function
from __future__ import division

import os
import sys

import tensorflow as tf

import input

batch_size = 30

IMAGE_SIZE = (224, 224, 3)
NUM_CLASSES = 3

LEARNING_RATE = 0.01

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = tf.get_variable(name=name, shape=shape,
            initializer = tf.truncated_normal_initializer(stddev=stddev,
                                                          dtype=tf.float32),
            dtype=tf.float32)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images):
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                          shape=[5,5,3,64],
                                          stddev=5e-2,
                                          wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [64], tf.float32,
                            tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

  pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                        padding='SAME', name='pool1')
  # TODO: Is this batch norm?
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                          shape=[5,5,64,64],
                                          stddev=5e-2,
                                          wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [64],
                              tf.float32,
                              tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name='pool2')

  with tf.variable_scope('local3') as scope:
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = tf.get_variable('biases', [384], tf.float32,
                            tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = tf.get_variable('biases', [192], tf.float32,
                            tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = tf.get_variable('biases', [NUM_CLASSES], tf.float32,
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases,
                            name=scope.name)

  return softmax_linear

def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(loss, learning_rate):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
