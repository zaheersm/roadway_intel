from __future__ import print_function

import sys
import os

from PIL import Image
import tensorflow as tf
import skimage.transform
import numpy as np

from context import settings

IMAGE_SIZE=298
IMAGE_SIZE_CROPPED=224

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

"""
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


def resize(im, size=224):
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

def crop_bbox(im, bbox):
  im = im [bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]]
  return im
"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
s = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                    gpu_options=gpu_options))

images, labels, bboxs = read_imagefile_label(settings.TEST_META)

images = tf.convert_to_tensor(images, dtype=tf.string)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)
bboxs = tf.convert_to_tensor(bboxs, dtype=tf.int32)
input_queue = tf.train.slice_input_producer([images, labels, bboxs],
                                          num_epochs=1,
                                          shuffle=False)
image, label, bbox = read_image(input_queue)
#image = tf.py_func(crop_bbox, [image, bbox], [tf.uint8])[0]
image = tf.image.crop_to_bounding_box(image, bbox[0], bbox[1],
                                           bbox[2], bbox[3])
image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE],
                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

#image = tf.py_func(resize, [image, IMAGE_SIZE], [tf.uint8])[0]

image = tf.random_crop(image, [IMAGE_SIZE_CROPPED,
                             IMAGE_SIZE_CROPPED, 3])
# Randomly flip the image horizontally.
image = tf.image.random_flip_left_right(image)

image = tf.image.random_brightness(image,
                                max_delta=0.5)
image = tf.image.random_contrast(image,
                               lower=0.7, upper=1.2)

image.set_shape((IMAGE_SIZE_CROPPED, IMAGE_SIZE_CROPPED,3))
image = tf.cast(image, tf.float32)
image = tf.cast(image, tf.uint8)
s.run(tf.initialize_all_variables())
s.run(tf.initialize_local_variables())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=s, coord=coord)
sample_count = 50
step = 0
for i in range (82690):
  step+=1
  im, l = s.run([image, label])
  print (step)
#im = Image.fromarray(im.astype(np.uint8), 'RGB')
#im.save(os.path.join(os.path.join(settings.PROJECT_ROOT,'samples'),
#                    str(l) + '.png'))
coord.request_stop()
coord.join(threads)
