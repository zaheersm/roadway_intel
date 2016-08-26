from __future__ import print_function

import os
import tensorflow as tf

from context import settings

def read_imagefile_label(imagefile_label):
  """Reads .txt files and returns lists of imagefiles paths and labels
  Args:
    imagefile_label: .txt file with image file paths and labels in each line
  Returns:
    imagefile: list with image file paths
    label: list with labels
  """
  f = open(imagefile_label)
  imagefiles = []
  labels = []
  for line in f:
    line = line[:-1].split(' ')
    im = line[0]
    l = line[1]
    imagefiles.append(im)
    labels.append(int(l))
  return imagefiles, labels

def read_image(input_queue):
  im = tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3)
  im = tf.cast(im, tf.float32)
  l = input_queue[1]
  return im, l

def inputs(train, batch_size=10, num_epochs=None):
  with tf.name_scope('input'):
    images, labels = read_imagefile_label(settings.FILENAME_LABEL)
    images = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)
    image, label = read_image(input_queue)
    # TODO: Improve this resize method since it downgrades the quality
    image = tf.image.resize_images(image, 112, 112, 
                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    print (image)
    print (label)
    # Convert from [0, 255] -> [-0.5, 0.5] floats
    image = image * (1. /255) - 0.5
    images, sparse_labels = tf.train.shuffle_batch([image, label], 
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=1000 + 3 * batch_size,
                                                  min_after_dequeue=1000)
    return images, sparse_labels                             
    