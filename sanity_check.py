"""
Reads first 10 examples from a .tfrecords files
and store them into .png file as a sanity check
"""
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import tensorflow as tf

import input

train_dir = 'data/car_type'
TRAIN_FILE = 'train_test.tfrecords'
VALIDATION_FILE = 'valid_test.tfrecords'

IMAGE_SIZE = 224
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
  image = tf.reshape(image, tf.pack([IMAGE_SIZE, IMAGE_SIZE, 3]))
  image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
  image = tf.cast(image, tf.float32)
  label = tf.cast(features['label'], tf.int32)

  return image, label

def main():
  with tf.Session() as sess:
    #filename = os.path.join(train_dir, TRAIN_FILE)
    #filename_queue = tf.train.string_input_producer([filename])
    image, label = input.distorted_inputs(False, 1, 4)
    image  = (image + 0.5) * 255
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      i = 0
      while not coord.should_stop():
        im, l = sess.run([image, label])
        im = im[0].astype('uint8')
        im = Image.fromarray(im, 'RGB')
        im.save('samples/d_' + str(i) + '_' + str(l[0]) + '.png')
        i+=1
    except tf.errors.OutOfRangeError:
      print ('Sanity Check complete')
    except Exception as e:
      print (str(e))
    finally:
      coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  main()