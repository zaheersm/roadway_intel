from __future__ import print_function

import sys
import os

from PIL import Image
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
  im = tf.image.decode_jpeg(tf.read_file(input_queue[0]))
  l = input_queue[1]
  bbox = input_queue[2]
  return im, l, bbox

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
  return im.astype(np.uint8)

def crop_bbox(im, bbox):
  im = im [bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
  return im

with tf.Session() as s:
  images, labels, bboxs = read_imagefile_label(settings.TRAIN_META + '.bb')

  images = tf.convert_to_tensor(images, dtype=tf.string)
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  bboxs = tf.convert_to_tensor(bboxs, dtype=tf.int32)
  input_queue = tf.train.slice_input_producer([images, labels, bboxs],
                                              num_epochs=1,
                                              shuffle=False)
  image, label, bbox = read_image(input_queue)
  #image = tf.image.crop_to_bounding_box(image, bbox[0], bbox[1],
  #                                             bbox[2], bbox[2])
  image = tf.py_func(crop_bbox, [image, bbox], [tf.uint8])[0]

  #image = tf.image.resize_images(image, 224, 224,
  #                               tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
  image = tf.py_func(resize, [image], [tf.uint8])[0]
  image = tf.cast(image, tf.float32)
  s.run(tf.initialize_all_variables())
  s.run(tf.initialize_local_variables())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  sample_count = 10
  for i in range (sample_count):
    im, l = s.run([image, label])
    print (im.shape)
    im = Image.fromarray(im.astype(np.uint8), 'RGB')
    im.save(os.path.join(settings.SAMPLES_DIR, str(l) + '.png'))
  coord.request_stop()
  coord.join(threads)
