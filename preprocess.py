from __future__ import print_function

import sys
import os
import unicodedata

import scipy.io
import numpy as np

from PIL import Image
import progressbar

from sklearn.cross_validation import StratifiedShuffleSplit
import skimage.transform

import tensorflow as tf

# Goal is to pre-process jpegs into TFRecord format
DIR = 'cars_dataset'
channels = 3
height = 224
width = 224

def get_label_mapping():
  meta = scipy.io.loadmat('cars_dataset/devkit/cars_meta.mat')
  class_name_raw = meta['class_names'][0]
  def ascii(unicode_str):
    return unicodedata.normalize('NFKD', unicode_str).encode('ascii', 'ignore')
  class_name = np.ndarray((196,), dtype = np.object)
  for i in range(len(class_name)):
    class_name[i] = ascii(class_name_raw[i][0])
  annotations = scipy.io.loadmat('cars_dataset/devkit/' + \
                                 'cars_train_annos.mat')['annotations'][0]
  N = len (annotations)
  label_file = {}
  for i in range(N):
    id = annotations[i][4][0][0]
    file_name = ascii(annotations[i][5][0])
    label_file[file_name] = id - 1
  
  return label_file

def prep_image(im):
    
  # Resize so smallest dim = 256, preserving aspect ratio
  h, w, _ = im.shape
  if h < w:
    im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
  else:
    im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

  # Central crop to 224x224
  h, w, _ = im.shape
  im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
  # Shuffle axes so we have channels*height*width
  im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
   
  return im.astype('float32')


def get_im_labels(label_file): 
  IMAGES_DIR = os.path.join(DIR, 'cars_train')
  image_files = os.listdir(IMAGES_DIR)
  N = len(image_files)
  images = np.ndarray((N, channels, height, width), dtype = np.float32)
  labels = np.ndarray((N,), dtype = np.uint8)

  current_image = 0
  discarded = 0
  bar = progressbar.ProgressBar(maxval=N, \
                      widgets=[progressbar.Bar('=','[',']'), ' ', 
                      progressbar.Percentage()])
  bar.start()
  for f in image_files:
    im = np.asarray(Image.open(os.path.join(IMAGES_DIR,f)))
    # Discarding the images which are not RGB (That is, they are grayscale)
    if len(im.shape) != 3:
        discarded+=1
        continue
    im  = prep_image(im)
    images[current_image, :, :, :] = im
    labels[current_image] = label_file[f]
    current_image+=1
    
    # Printing the progress
    bar.update(current_image)
  bar.finish()
  return images, labels

def split(images, labels):
  sss = StratifiedShuffleSplit(labels, 1, 0.2)
  for train_index, test_index in sss:
    train_im = images[train_index]
    train_label = labels[train_index]
    test_im = images[test_index]
    test_label = labels[test_index]
  
  # Further spltting test into test and validation sets
  sss = StratifiedShuffleSplit(test_label, 1, 0.5)
  for valid_index, test_index in sss:
    valid_im = test_im[valid_index]
    valid_label = test_label[valid_index]
    test_im = test_im[test_index]
    test_label = test_label[test_index]

  return train_im, train_label, valid_im, valid_label, test_im, test_label 

def convert_TFR(dataset, name):
  images = dataset[0]
  labels = dataset[1]
  assert len(images) == len(labels)
  num_examples = len(images)
  rows = images.shape[2]
  cols = images.shape[3]
  depth = images.shape[1]

  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  filename = os.path.join(DIR, name + '.tfrecords')
  print ('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()



def main():
  label_file = get_label_mapping()
  print ('Reading images into numpy arrays')
  images, labels = get_im_labels(label_file)
  print (images.shape)
  print (labels.shape) 
  print ('Preparing train, validation and test sets')
  train_im, train_label, valid_im, \
            valid_label, test_im, test_label = split(images, labels)
  
  print ('Storing dataset into TFRecords file format')
  convert_TFR((train_im, train_label), 'train')
  convert_TFR((valid_im, valid_label), 'valid')
  convert_TFR((test_im, test_label), 'test')   

if __name__ == '__main__':
  main()
