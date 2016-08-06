
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys

import numpy as np
import tensorflow as tf

import mlp
import input

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                           """Directory where to read model checkpoints.""")

def evaluate():

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    images, labels = input.inputs(train=False, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)
    # Build a Graph that computes predictions from the inference model.
    logits = mlp.inference(images,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    saver = tf.train.Saver(tf.trainable_variables())

    init = tf.initialize_all_variables()
    # Create a session for running operations in the Graph.
    sess = tf.Session()
    sess.run(init)
     
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      print (ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    true_count = 0
    total_count = 0
    try:
      step = 0
      while not coord.should_stop():

        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        total_count += 100
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      print ('Precision: %f' % (true_count/total_count))
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

def main():
  evaluate()


if __name__ == '__main__':
  main()

