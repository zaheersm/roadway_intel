from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys

import numpy as np
import tensorflow as tf

import lenet
import input

# Defining basic model parameters
learning_rate = 0.0003
num_epochs = 1
batch_size = 30
checkpoint_dir = 'checkpoints'

def run_training():

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    with tf.device('/gpu:0'):
      images, labels = input.inputs(train=True, batch_size=batch_size,
                              num_epochs=num_epochs)
      # Build a Graph that computes predictions from the inference model.
      logits = lenet.inference(images)

      # Add to the Graph the loss calculation.
      loss = lenet.loss(logits, labels)

      # Add to the Graph operations that train the model.
      train_op = lenet.train(loss, learning_rate)

      # The op for initializing the variables.
      init_op = tf.group(tf.initialize_all_variables(),
                         tf.initialize_local_variables())
    
    saver = tf.train.Saver(tf.trainable_variables())
    # Create a session for running operations in the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    sess.run(init_op)
    step = 0

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print ('Restoring from checkpoint %s' % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      print ('No checkpoint file found - Starting training from scratch')

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
        if step % 50 == 0:
          checkpoint_path = os.path.join('checkpoints','model.ckpt')
          saver.save(sess, checkpoint_path,global_step=step)
    except tf.errors.OutOfRangeError:
      print ('tf.errors.OutOfRangeError')
    finally:
      print('Done training for %d steps.' % (step))
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

def main():
  run_training()

if __name__ == '__main__':
  main()
