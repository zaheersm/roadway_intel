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
import evaluate

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', None, 'Number of epochs to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints',
                           """Directory where to read model checkpoints.""")

def run_training():

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    images, labels = input.inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)
    # Build a Graph that computes predictions from the inference model.
    logits = mlp.inference(images,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the loss calculation.
    loss = mlp.loss(logits, labels)

    # Add to the Graph operations that train the model.
    train_op = mlp.training(loss, FLAGS.learning_rate)

    # The op for initializing the variables.
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    
    saver = tf.train.Saver(tf.trainable_variables())
    # Create a session for running operations in the Graph.
    sess = tf.Session()

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)
    step = 0
    epoch = 0


    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      print ('Restoring from checkpoint %s' % ckpt.model_checkpoint_path)
      saver.restore(sess, ckpt.model_checkpoint_path)
      step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      epoch = step / 65
    else:
      print ('No checkpoint file found - Starting training from scratch')

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    patience = 650 # 3 epochs
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = 65
    best_validation_loss = np.inf
    test_score = 0.
    done_looping = False
    epoch = 0
    n_train_batches = 65 # We have 6500 training examples and batchsize of 100

    try:
      while not done_looping and not coord.should_stop():
        epoch = epoch + 1
        
        for minibatch_index in range (n_train_batches):
          start_time = time.time()

          # Run one step of the model.  The return values are
          # the activations from the `train_op` (which is
          # discarded) and the `loss` op.  To inspect the values
          # of your ops or variables, you may include them in
          # the list passed to sess.run() and the value tensors
          # will be returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss])
          duration = time.time() - start_time
          step += 1
          # Print an overview fairly often.
          if step % 65 == 0:
            print('Epoch %d | Step %d: loss = %.2f (%.3f sec)' % (epoch,
                                                    step, 
                                                    loss_value,
                                                    duration))
            checkpoint_path= os.path.join('checkpoints','model.ckpt')
            saver.save(sess, checkpoint_path,global_step=step)
            val_loss = evaluate.eval()
            print ('Validation loss = %.2f' % val_loss)            
            if val_loss < best_validation_loss:
              if val_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, step*patience_increase)
                best_validation_loss = val_loss
                saver.save(sess, os.path.join('checkpoints', 'best_model'))
          if patience <= step:
            done_looping = True
            break  

            
    
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main():
  run_training()


if __name__ == '__main__':
  main()
