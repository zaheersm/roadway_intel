from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from context import settings

train_dir = os.path.join(settings.DATA_ROOT, 'car_type')
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'valid_os.tfrecords'
TEST_FILE = 'test_os.tfrecords'

IMAGE_SIZE = 224
IMAGE_SIZE_CROPPED = 168 # 224 x 0.75 = 168

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length IMAGE_PIXELS) to a tf.float32 tensor with shape
  # [IMAGE_SIZE, IMAGE_SIZE, 3].
  image = tf.decode_raw(features['image_raw'], tf.float32)
  image = tf.reshape(image, tf.pack([IMAGE_SIZE, IMAGE_SIZE, 3]))
  image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

  image = tf.cast(image, tf.float32)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)
  return image, label

def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, NUM_CLASSES).
    Note that a tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    # Image processing for evaluation.
    # Crop the central [IMAGE_SIZE_CROPPED, IMAGE_SIZE_CROPPED] of the image.
    image = tf.image.resize_image_with_crop_or_pad(image,
                                                   IMAGE_SIZE_CROPPED,
                                                   IMAGE_SIZE_CROPPED)
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = image * (1. / 255) - 0.5

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=64,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels

def distorted_inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Images before being passed to training routines are distorted with random
  crop, flip, brightness, contrast and whitening

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, NUM_CLASSES).
    Note that a tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    distorted_image = tf.random_crop(image, [IMAGE_SIZE_CROPPED,
                                             IMAGE_SIZE_CROPPED, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=25)
    distorted_image = tf.image.random_contrast(distorted_image,
                                              lower=0.7, upper=1.2)
    #Convert from [0, 255] -> [-0.5, 0.5] floats.
    distorted_image = distorted_image * (1. / 255) - 0.5

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    distorted_images, sparse_labels = tf.train.shuffle_batch(
        [distorted_image, label], batch_size=batch_size, num_threads=64,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return distorted_images, sparse_labels
