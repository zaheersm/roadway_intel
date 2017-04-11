from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import roadway as rd

def _get_learning_rate(base_learning_rate, global_step,
                       decay_steps, decay_factor):
  """ Helper to get learning_rate
  """
  return tf.train.exponential_decay(base_learning_rate, global_step,
                                    decay_steps, decay_factor,
                                    staircase=True)

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
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def run_training(no_classes, batch_size, epochs, steps_per_epoch,
                 base_learning_rate, decay_steps, decay_factor, 
                 no_gpus, checkpoint_dir):
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False)
    
    
    #Using the same learning rate for fine-tuning softmax, fcs and convs.
    #Separate learning rate could be used by just plugging in the values
    
    decay_steps = decay_steps * 3
    lr_softmax = _get_learning_rate(base_learning_rate, global_step,
                                    decay_steps, decay_factor)
    lr_fc = _get_learning_rate(base_learning_rate, global_step,
                               decay_steps, decay_factor)
    lr_conv = _get_learning_rate(base_learning_rate, global_step,
                                 decay_steps, decay_factor)
    op1 = tf.train.GradientDescentOptimizer(lr_softmax)
    op2 = tf.train.GradientDescentOptimizer(lr_fc)
    op3 = tf.train.GradientDescentOptimizer(lr_conv)
  
    tower_softmax_grads = []
    tower_fcs_grads = []
    tower_convs_grads = []

    images, labels = rd.input.distorted_inputs(True, batch_size, epochs)
    split_images = tf.split(images, no_gpus, 0)
    split_labels = tf.split(labels, no_gpus, 0)
    loss = []
    for i in xrange(no_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('tower',i)) as scope:
          logits = rd.vgg16.model.inference(split_images[i], no_classes,
                                            keep_prob=0.5)
          loss.append(rd.vgg16.model.loss_function(logits, split_labels[i]))

          tf.get_variable_scope().reuse_variables()

          trainable_vars = tf.trainable_variables()
          convs = trainable_vars[20:26]
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
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                            gpu_options=gpu_options))
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
      rd.vgg16.model.load_weights('vgg16_weights.npz', sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
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
        # Epoch
        if step % steps_per_epoch == 0:
          print ('Saving Model')
          checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
      print ('tf.errors.OutOfRangeError')
    finally:
      print('Done training for %d steps.' % (step))
      # When done, ask the threads to stop.
      coord.request_stop()
    coord.join(threads)
    sess.close()
