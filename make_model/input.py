from __future__ import print_function

import os
import tensorflow as tf
import skimage.transform
import numpy as np

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

def resize(im):
  # Resize so smallest dim = 256, preserving aspect ratio
  h, w, _ = im.shape
  if h < w:
    im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
  else:
    im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
  # Central crop to 224x224
  h, w, _ = im.shape
  im = im[h//2-112:h//2+112, w//2-112:w//2+112]
  return im.astype(np.float32)

def inputs(train, batch_size=10, num_epochs=None):
  with tf.name_scope('input'):
    if train == True:
      images, labels = read_imagefile_label(settings.TRAIN_META)
    else:
      images, labels = read_imagefile_label(settings.VALID_META)

    images = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True)
    image, label = read_image(input_queue)
    #image = tf.image.resize_images(image, 224, 224,
    #                               tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    image = tf.py_func(resize, [image], [tf.float32])[0]
    # Spits out error if I don't explicitly set shape
    # There should be more natural way to it
    image.set_shape((224,224,3))

    mean = tf.constant([123.86, 116.779, 103.939], 
                        dtype=tf.float32, 
                        shape=[1,1,3])
    image = image - mean
    images, sparse_labels = tf.train.shuffle_batch([image, label], 
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=1000+3*batch_size,
                                                  min_after_dequeue=1000)
    return images, sparse_labels
