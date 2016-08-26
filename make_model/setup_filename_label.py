from __future__ import print_function

import sys
import os
import shutil
import operator
import numpy as np
import scipy.io

from context import settings

root = os.path.join(settings.DATA_ROOT, 'car_type')
images = os.path.join(settings.DATA_ROOT, 'image')
labels = os.path.join(settings.DATA_ROOT, 'label')
attributes = os.path.join(settings.DATA_ROOT, 'misc/attributes.txt')

f = open(settings.FILENAME_LABEL, 'w')

no_models = 0
labels_dict = {}
image_count = 0
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
        f.write('%s %d %s %s\n' % (os.path.join(image_make_model_year, image),
                                        labels_dict[model],
                                        make, model))
        image_count+=1
print ('#Models: %d' % no_models) 
print ('#TotalImages: %d' % image_count) 
