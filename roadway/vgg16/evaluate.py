from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import tensorflow as tf

import roadway as rd

def run_evaluation(no_classes, batch_size, checkpoint_dir, k=5,gpu_id=2):

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    with tf.device('/gpu:%d' % gpu_id):
      images, labels = rd.input.inputs(train=False, batch_size=batch_size,
                                       num_epochs=1)
      # Build a Graph that computes predictions from the inference model.
      logits = rd.vgg16.model.inference(images, no_classes, keep_prob=1.0)

      # Add to the Graph the loss calculation.
      loss = rd.vgg16.model.loss_function(logits, labels)
      top_k_op = tf.nn.in_top_k(logits, labels, k)
    
    # To restore the latest checkpoint for evaluation
    saver = tf.train.Saver(tf.trainable_variables())

    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    # Create a session for running operations in the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    sess.run(init)
     
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      print ('Checkpoint: %s' % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      sys.exit(-1)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    true_count = 0
    total_count = 0
    total_loss = 0.
    try:
      step = 0
      while not coord.should_stop():
        predictions, loss_value, labels_value = sess.run([top_k_op, loss, labels])
        #analyze(predictions, labels_value)
        true_count += np.sum(predictions)
        total_count += batch_size
        total_loss+=loss_value
        step += 1
    except tf.errors.OutOfRangeError:
      print('Evaluation Complete ')
    except Exception as e:
      print (str(e))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      precision = true_count/total_count
      print ('Precision: %f Steps: %d' % (precision, step))
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    avg_loss = total_loss/step
    print ('Eval loss: %.2f' % avg_loss)
    return avg_loss
