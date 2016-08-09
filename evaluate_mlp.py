
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

# Defining basic model parameters
learning_rate = 0.01
num_epochs = None
hidden1 = 128
hidden2 = 32
batch_size = 100
checkpoint_dir = 'checkpoints'

def eval():

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    

    # Input images and labels.
    images, labels = input.inputs(train=False, batch_size=batch_size,
                            num_epochs=1)
    images = tf.reshape(images, [batch_size, -1])
    # Build a Graph that computes predictions from the inference model.
    logits = mlp.inference(images,
                             hidden1,
                             hidden2)

    # Add to the Graph the loss calculation.
    loss = mlp.loss(logits, labels)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    
    # To restore the latest checkpoint for evaluation
    saver = tf.train.Saver(tf.trainable_variables())

    init = tf.initialize_all_variables()
    # Create a session for running operations in the Graph.
    sess = tf.Session()
    sess.run(init)
     
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      print ('Evaluating: %s' % ckpt.model_checkpoint_path)
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
    total_loss = 0.
    try:
      step = 0
      while not coord.should_stop():

        predictions, loss_value = sess.run([top_k_op, loss])
        true_count += np.sum(predictions)
        total_count += 100
        step += 1
        total_loss+=loss_value
    except tf.errors.OutOfRangeError:
      print('Evaluation Complete ')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      precision = true_count/total_count
      print ('Precision: %f Steps: %d' % (precision, step))
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    return total_loss/step

def main():
  eval()


if __name__ == '__main__':
  main()

