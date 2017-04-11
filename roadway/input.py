from __future__ import print_function

import os
import tensorflow as tf
import skimage.transform
import numpy as np
import sys

import settings

IMAGE_SIZE = 298
IMAGE_SIZE_CROPPED = 224

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
  bboxs = []
  for line in f:
    line = line[:-1].split(' ')
    im = line[0]
    l = line[1]
    bbox = line[2:]
    for idx, value in enumerate(bbox):
      bbox[idx] = int(bbox[idx])
    imagefiles.append(im)
    labels.append(int(l))
    bboxs.append(bbox)
  return imagefiles, labels, bboxs

def read_image(input_queue):
  im = tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3)
  l = input_queue[1]
  bbox = input_queue[2]
  return im, l, bbox

""""
def crop_bbox(im, bbox):
  im = im [bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
  return im

def resize(im, size):
  # Resize so smallest dim = size+2, preserving aspect ratio
  h, w, _ = im.shape
  if h < w:
    im = skimage.transform.resize(im, ((size+2), w*(size+2)/h), preserve_range=True)
  else:
    im = skimage.transform.resize(im, (h*(size+2)/w, (size+2)), preserve_range=True)
  # Central crop to sizexsize
  h, w, _ = im.shape
  im = im[h//2-(size/2):h//2+(size/2), w//2-(size/2):w//2+(size/2)]
  return im.astype(np.uint8)
"""

def inputs(train, batch_size=10, num_epochs=None):
  with tf.name_scope('input'):
    if train == True:
      images, labels, bboxs = read_imagefile_label(settings.TRAIN_META)
    else:
      images, labels, bboxs = read_imagefile_label(settings.TEST_META)

    images = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    bboxs = tf.convert_to_tensor(bboxs, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels, bboxs],
                                                num_epochs=num_epochs,
                                                shuffle=False)
    image, label, bbox = read_image(input_queue)
    #image = tf.py_func(crop_bbox, [image, bbox], [tf.uint8])[0]
    try:
        image = tf.image.crop_to_bounding_box(image, bbox[0], bbox[1],
                                                 bbox[2], bbox[3])
    except:
        print ("\n\n",bbox[0], bbox[1], bbox[2], bbox[3],"\n\n")
    image = tf.image.resize_images(image,
                                   [IMAGE_SIZE_CROPPED,
                                   IMAGE_SIZE_CROPPED],
                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    #image = tf.py_func(resize, [image, IMAGE_SIZE_CROPPED], [tf.uint8])[0]
    image = tf.cast(image, tf.float32)
    # Spits out error if I don't explicitly set shape
    # There should be more natural way to it
    image.set_shape((IMAGE_SIZE_CROPPED, IMAGE_SIZE_CROPPED,3))

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

def distorted_inputs(train, batch_size=10, num_epochs=None):
  with tf.name_scope('input'):
    if train == True:
      images, labels, bboxs = read_imagefile_label(settings.TRAIN_META)
    else:
      images, labels, bboxs = read_imagefile_label(settings.TEST_META)

    images = tf.convert_to_tensor(images, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    bboxs = tf.convert_to_tensor(bboxs, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels, bboxs],
                                                num_epochs=num_epochs,
                                                shuffle=False)
    image, label, bbox = read_image(input_queue)
    #image = tf.py_func(crop_bbox, [image, bbox], [tf.uint8])[0]
    try:
        image = tf.image.crop_to_bounding_box(image, bbox[0], bbox[1],
                                                 bbox[2], bbox[3])
    except tf.errors.InvalidArgumentError:
        print ("\n\n\n",bbox[0], bbox[1], bbox[2], bbox[3], "\n\n\n")
    except:
        print ("\n\n\n",bbox[0], bbox[1], bbox[2], bbox[3],"\n\n\n")
    image = tf.image.resize_images(image,
                                   [IMAGE_SIZE,
                                   IMAGE_SIZE],
                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    #image = tf.py_func(resize, [image, IMAGE_SIZE_CROPPED], [tf.uint8])[0]

    image = tf.random_crop(image, [IMAGE_SIZE_CROPPED,
                                   IMAGE_SIZE_CROPPED, 3])
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_contrast(image,
                                   lower=0.7, upper=1.2)
    # Spits out error if I don't explicitly set shape
    # There should be more natural way to it
    image.set_shape((IMAGE_SIZE_CROPPED, IMAGE_SIZE_CROPPED,3))

    image = tf.cast(image, tf.float32)
    mean = tf.constant([123.86, 116.779, 103.939],
                        dtype=tf.float32,
                        shape=[1,1,3])
    image = image - mean
    images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=16,
                                                  capacity=1000+4*batch_size,
                                                  min_after_dequeue=1000)
    return images, sparse_labels
