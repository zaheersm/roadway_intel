"""
data_preparation.py does the following key steps:
1. Determines car_type of each model_id from data/misc/attributes.txt file
2. Creates data/car_type directory with car_types (e.g. 1, 2, ... 12) as subdirectories
3. Copies images from data/image/ to data/car_type/ depending upon the car_type
4. Reads up images names and prepare train/valid/test sets
5. Write images and labels into train/valid/test TFrecords format 
   (e.g. data/car_type/train.tfrecords etc.)

Pre-requisites: CompCars dataset uzipped at data/

This script would take ~1-1.5hrs to complete so you might want to 
grab a cup of coffee
"""
from __future__ import print_function

import sys
import os
import shutil

import numpy as np
import scipy.io

from PIL import Image
from sklearn.cross_validation import StratifiedShuffleSplit
import skimage.transform

import progressbar

import tensorflow as tf

root = 'data/car_type'
images = 'data/image'
labels = 'data/label'
attributes = 'data/misc/attributes.txt'

path_suv = '2'
path_sedan = '3'
path_hatchback = '4'



"""
  There are total 1716 model ids which are not contiguous
  (i.e. model ids span from 1 to 2004)
  Returns two dictionaries:
  1. model_id (1-2004) -> contiguous model ids (0 - 1715)
  2. contiguous model ids to car-type (e.g. SUV, Sedan, hatchback etc.)
"""
def get_mapping():
  unknown = 0
  f = open(attributes)
  content = f.read().split('\n')
  content = content[1: -1]
  cont_models = {}
  car_type = {}
  
  N = len(content)

  for idx in range(N):
    cont_models[content[idx].split(' ')[0]] = idx
    car_type[idx] = int(content[idx].split(' ')[-1])
    if car_type[idx] == 0:
      unknown+=1
  return N, unknown, cont_models, car_type

def setup_directories():
  if os.path.exists(root):
    shutil.rmtree(root)
  os.makedirs(root)

  for i in range(1, 13):
    d = os.path.join(root, str(i))
    os.makedirs(d)  

  return root

def copy():
  N, unknown, cont_models, car_type = get_mapping()

  print ('Total Models: %d, Models with unknown type: %d' % (N, unknown))
  print ('Copying images from data/image to data/car_type')
  root = setup_directories()
  bar = progressbar.ProgressBar(maxval=N, \
                        widgets=[progressbar.Bar('=','[',']'), ' ', 
                        progressbar.Percentage()])
  bar.start()

  # count variable updates the progress-bar
  count = 0
  makes =  os.listdir(images)
  for make in makes:
    image_make = os.path.join(images, make)
    models = os.listdir(image_make)
    for model in models:
      image_make_model = os.path.join(image_make, model)
      count+=1
      bar.update(count)
      if car_type[cont_models[model]] == 0:
        # Car type is not known, let's ignore
        continue
      years = os.listdir(image_make_model)
      for year in years:
        image_make_model_year = os.path.join(image_make_model, year)
        image_files = os.listdir(image_make_model_year)
        for image in image_files:
          dst = os.path.join(root, str(car_type[cont_models[model]]))
          shutil.copy(os.path.join(image_make_model_year, image), dst)
  bar.finish()

def to_np (l):
  ary  = np.ndarray((len(l),), dtype = np.object)
  for i in range(len(l)):
    ary[i] = l[i]
  return ary

def split(arr):
  N = len(arr)
  labels = np.zeros((N,))
  sss = StratifiedShuffleSplit(labels, 1, 0.2)
  for train_index, test_index in sss:
    train = arr[train_index]
    test = arr[test_index]
  
  N = len(test)
  labels = np.zeros((N,))
  sss = StratifiedShuffleSplit(labels, 1, 0.5)
  for valid_index, test_index in sss:
    valid = test[valid_index]
    test = test[test_index]

  return train, valid, test

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

def squash (ar0, ar1, ar2):
  l0, l1, l2 = len(ar0), len(ar1), len(ar2)
  n = l0 + l1 + l2
  im = np.ndarray((n,), dtype = np.object)
  label = np.ndarray((n,), dtype = np.uint8)
  
  im[0: l0] = ar0[:]
  label[0: l0] = 0
  im[l0: l0+l1] = ar1[:]
  label[l0: l0+l1] = 1
  im[l0+l1: l0+l1+l2] = ar2[:]
  label[l0+l1: l0+l1+l2] = 2

  return im, label

def shuffle(im, label):
  assert len(im) == len(label)
  p = np.random.permutation(len(im))
  return im[p], label[p]

lpath = {0: '2', 1:'3', 2:'4'}

def process_TFR(images, labels, name):
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  assert len(images) == len(labels)
  num_examples = len(images)

  filename = os.path.join(root, name + '.tfrecords')
  print ('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  

  bar = progressbar.ProgressBar(maxval=num_examples, \
                      widgets=[progressbar.Bar('=','[',']'), ' ', 
                      progressbar.Percentage()])
  bar.start()
  
  written = 0
  scraped = 0
  for index in range(num_examples):
    l = labels[index]
    DIR = os.path.join(root, lpath[l])

    im = np.asarray(Image.open(os.path.join(DIR, images[index])))
    assert len(im.shape) == 3
    im = prep_image(im)
    rows, cols, depth = im.shape[1], im.shape[2], im.shape[0]
    if depth != 3:
      scraped+=1
      continue #Skip the example if it doesn't have 3 channels

    assert rows == 224 and cols == 224 and depth == 3
    
    image_raw = im.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    written+=1
    bar.update(index)
  writer.close()
  bar.finish()
  print ('Examples written: %d | scraped: %d' % (written, scraped))

def main():
  # Copying data from data/images to data/car_types/ as per the car types
  copy()
  
  suv = (os.listdir(os.path.join(root, path_suv)))
  sedan = (os.listdir(os.path.join(root, path_sedan)))
  hatchback = (os.listdir(os.path.join(root, path_hatchback)))

  suv = to_np(suv)
  sedan = to_np(sedan)
  hatchback = to_np(hatchback)

  suv_train, suv_valid, suv_test = split(suv)
  sedan_train, sedan_valid, sedan_test = split(sedan)
  hatchback_train, hatchback_valid, hatchback_test = split(hatchback)

  # Squashing suv, sedan and hatchback into training/valid and test sets
  train_im, train_label = squash(suv_train, sedan_train, hatchback_train)
  valid_im, valid_label = squash(suv_valid, sedan_valid, hatchback_valid)
  test_im, test_label = squash(suv_test, sedan_test, hatchback_test)

  train_im, train_label = shuffle (train_im, train_label)
  valid_im, valid_label = shuffle (valid_im, valid_label)
  test_im, test_label = shuffle (test_im, test_label)
  process_TFR(train_im, train_label, 'train')
  process_TFR(valid_im, valid_label, 'valid')
  process_TFR(test_im, test_label, 'test')


if __name__ == "__main__":
  main()




