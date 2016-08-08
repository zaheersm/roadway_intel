"""
Reads first 10 examples from a .tfrecords files
and store them into .png file as a sanity check
"""
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import tensorflow as tf

train_dir = 'cars_dataset'
TRAIN_FILE = 'train_test.tfrecords'
VALIDATION_FILE = 'valid.tfrecords'

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.float32)
  image = tf.reshape(image, tf.pack([3, 224, 224]))
  image.set_shape([3,224,224])
  label = tf.cast(features['label'], tf.int32)

  return image, label

def main():
  with tf.Session() as sess:
    filename = os.path.join(train_dir, TRAIN_FILE)
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue)
    image.set_shape([3, 224, 224])
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
      im, l = sess.run([image, label])
      im = np.swapaxes(np.swapaxes(im, 1, 0), 2, 1)
      im = im.astype('uint8')
      im = Image.fromarray(im, 'RGB')
      im.save('samples/' + TRAIN_FILE.split('.')[0] + '_' + str(i) + '_' + str(l) + '.png')
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  main()