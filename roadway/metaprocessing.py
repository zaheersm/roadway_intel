from __future__ import print_function

import sys
import os

import numpy as np
import sklearn.cross_validation

import settings

def _get_bbox(img_path):
  """ Helper Method to retrieve bounding boxe for a given image file
  """
  img_path = img_path.replace("image", "label")
  bbox_path = img_path.replace(".jpg", ".txt")
  f = open(bbox_path, 'r')
  for idx, line in enumerate(f):
    if idx == 2: #bbox coords are on the 3rd line
      coords = line.split(" ")
      for j, coord in enumerate(coords):
        coords[j] = int(coord) - 1 # Compensating for MATLAB 1-based indexing
  # bbox: [offset_height, offset_width, target_height, target_width]
  bbox = [coords[1], coords[0], coords[3]-coords[1], coords[2]-coords[0]]
  return bbox

def setup_meta():
  print ('Setting up meta-file')
  # Setting up paths for images, labels
  images = os.path.join(settings.DATA_ROOT, 'image')
  labels = os.path.join(settings.DATA_ROOT, 'label')

  # Counter for number of models (no_classes)
  no_models = 0
  # Counter for total number of images (training examples)
  image_count = 0 

  # Dictionary to map model ID to class_label
  labels_dict = {}

  # List for all image files (paths)
  imagefiles = []
  # Labels for all image files
  labels = []
  # List of bounding Boxes for all image files 
  # Each Entry: [offset_height, offset_width, target_height, target_width]
  bboxs = []

  makes = os.listdir(images)
  for make in makes:
    image_make = os.path.join(images, make)
    models = os.listdir(image_make)
    for model in models:
      image_make_model = os.path.join(image_make, model)
      no_images = 0
      for _, _, files in os.walk(image_make_model):
        no_images += len(files)
      if no_images < 100:
        continue # Ignore this model
      # Add this model to labels dict 
      labels_dict[model] = no_models
      no_models+=1
      years = os.listdir(image_make_model)
      for year in years:
        image_make_model_year = os.path.join(image_make_model, year)
        image_files = os.listdir(image_make_model_year)
        for image in image_files:
          imagefile = os.path.join(image_make_model_year, image)
          bbox = _get_bbox(imagefile)
          if bbox[2] <= 0 or bbox[3] <= 0:
            continue
          imagefiles.append(imagefile)
          bboxs.append(bbox)
          labels.append(labels_dict[model])
          image_count+=1

  imagefiles = np.array(imagefiles, dtype=np.object)
  labels = np.array(labels, dtype=np.int32)
  bboxs = np.array(bboxs)
  
  # Perform Train/Valid/Test Split using Stratified Split
  sss = sklearn.cross_validation.StratifiedShuffleSplit(labels, 1, 0.2)
  for train_index, test_index in sss:
      imagefiles_train = imagefiles[train_index]
      labels_train = labels[train_index]
      bboxs_train = bboxs[train_index]
      imagefiles_test = imagefiles[test_index]
      labels_test = labels[test_index]
      bboxs_test = bboxs[test_index]
  # Further spltting test into test and validation sets
  sss = sklearn.cross_validation.StratifiedShuffleSplit(labels_test, 1, 0.5)
  for valid_index, test_index in sss:
      imagefiles_valid = imagefiles_test[valid_index]
      labels_valid = labels_test[valid_index]
      bboxs_valid = bboxs_test[valid_index]
      imagefiles_test = imagefiles_test[test_index]
      labels_test = labels_test[test_index]
      bboxs_test = bboxs_test[test_index]

  # The total number of samples should be in multiple of 10s
  # to avoid nasty queuing errors/warnings later
  N = (len(imagefiles_train)/10)*10
  imagefiles_train = imagefiles_train[:N]
  labels_train = labels_train[:N]
  N = (len(imagefiles_valid)/10)*10
  imagefiles_valid = imagefiles_valid[:N]
  labels_valid = labels_valid[:N]
  N = (len(imagefiles_test)/10)*10
  imagefiles_test = imagefiles_test[:N]
  labels_test = labels_test[:N]

  # Writing Meta File for Training Data
  f = open(settings.TRAIN_META, 'w')
  for idx in range(len(imagefiles_train)):
    f.write('%s %d %d %d %d %d\n' % (imagefiles_train[idx], labels_train[idx],
                                     bboxs_train[idx][0], bboxs_train[idx][1],
                                     bboxs_train[idx][2], bboxs_train[idx][3]))

  # Writing Meta File for Valid Data
  f = open(settings.VALID_META, 'w')
  for idx in range(len(imagefiles_valid)):
    f.write('%s %d %d %d %d %d\n' % (imagefiles_valid[idx], labels_valid[idx],
                         bboxs_valid[idx][0], bboxs_valid[idx][1],
                         bboxs_valid[idx][2], bboxs_valid[idx][3]))

  # Writing Meta File for Test Data
  f = open(settings.TEST_META, 'w')
  for idx in range(len(imagefiles_test)):
    f.write('%s %d %d %d %d %d\n' % (imagefiles_test[idx], labels_test[idx],
                         bboxs_test[idx][0], bboxs_test[idx][1],
                         bboxs_test[idx][2], bboxs_test[idx][3]))
  print ('#Models: %d' % no_models) 
  print ('#TotalImages: %d' % image_count)
  print ('#TrainImages: %d' % len(imagefiles_train))
  print ('#ValidImages: %d' % len(imagefiles_valid))
  print ('#TestImages: %d' % len(imagefiles_test))
  
  return no_models, len(imagefiles_train),\
         len(imagefiles_valid), len(imagefiles_test)

def get_no_classes():
  # Reading the training_meta file and determining the max label
  labels = []
  f = open(settings.TRAIN_META)
  for line in f:
    labels.append(int(line.split(" ")[1]))
  return max(labels) + 1

def get_no_training_samples():
  return sum(1 for line in open(settings.TRAIN_META))

if __name__ == '__main__':
  setup_meta()
