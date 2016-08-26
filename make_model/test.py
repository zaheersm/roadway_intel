from __future__ import print_function

import sys
import os

from PIL import Image
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
  im = tf.image.decode_jpeg(tf.read_file(input_queue[0]))
  l = input_queue[1]
  return im, l

with tf.Session() as s:
  images, labels = read_imagefile_label(settings.FILENAME_LABEL)

  images = tf.convert_to_tensor(images, dtype=tf.string)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)

  input_queue = tf.train.slice_input_producer([images, labels],
                                              num_epochs=1,
                                              shuffle=True)
  image, label = read_image(input_queue)
  image = tf.image.resize_images(image, 224, 224, 
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  s.run(tf.initialize_all_variables())
  s.run(tf.initialize_local_variables())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  
  sample_count = 10
  for i in range (sample_count):
    im, l = s.run([image, label])
    print (im.shape)
    im = Image.fromarray(im, 'RGB')
    im.save(os.path.join(settings.SAMPLES_DIR, str(l) + '.png'))
  coord.request_stop()
  coord.join(threads)
  
