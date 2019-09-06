import logging
import sys
import os
import numpy as np
from PIL import Image
import math
import time
import random
from keras.preprocessing.image import ImageDataGenerator

from data_preprocessing import Wound


# logging setting
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('image_preprocessing.log', 'w', 'utf-8'), ])

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)


data_dir = 'data'
vgg_path = os.path.join(data_dir, 'vgg16.npy')
wound_dir = os.path.join(data_dir, 'wound')
wound_training_dir = os.path.join(wound_dir, 'training')
wound_validation_dir = os.path.join(wound_dir, 'validation')
wound_test_dir = os.path.join(wound_dir, 'test')
original_dir = 'origin'
truth_dir = 'ground_truth'

wound_ori_original_dir = os.path.join(wound_dir, 'original_training_data')
wound_ori_truth_dir = os.path.join(wound_dir, 'original_ground_truth')
inverted_truth_wound_dir = os.path.join(wound_dir, 'inverted_ground_truth')
resized_training_wound_dir = os.path.join(wound_dir, 'resized_training_data')
wound_training_original_dir = os.path.join(wound_training_dir, original_dir)
wound_training_truth_dir = os.path.join(wound_training_dir, truth_dir)


# wound = Wound(wound_ori_original_dir, wound_ori_truth_dir)
wound = Wound(resized_training_wound_dir, inverted_truth_wound_dir)

# # invert and resize the ground truth
# invert_imgs, invert_names = Wound.invert_imgs(wound.truth_imgs, wound.truth_names, inverted_truth_wound_dir, change_range = True)
# Wound.size_mag_imgs(invert_imgs, invert_names, 0.5, inverted_truth_wound_dir)
# logging.info('inverting ground truth is completed')

# # resize the training data (*0.5)
# Wound.size_mag_imgs(wound.ori_imgs, wound.ori_names, 0.5, resized_training_wound_dir)
# logging.info('resizing training data is completed')
# exit()

datagen = ImageDataGenerator(rotation_range = 30,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2)

augmentation_dir = 'data/wound/augmentation'
augmentation_ori_dir = os.path.join(augmentation_dir, original_dir)
augmentation_truth_dir = os.path.join(augmentation_dir, truth_dir)

total_count = 120
seed = int(random.random() * total_count)
logging.info('seed = %d' % seed)

# gen1 = datagen.flow(wound.ori_imgs, batch_size = 1, seed = seed, shuffle = False, save_to_dir = augmentation_dir, save_prefix = 'ori_', save_format = 'jpeg')
# gen2 = datagen.flow(wound.truth_imgs, batch_size = 1, seed = seed, shuffle = False, save_to_dir = augmentation_dir, save_prefix = 'truth_', save_format = 'jpeg')

gen1 = datagen.flow(wound.ori_imgs, batch_size = 1, seed = seed, shuffle = False, save_to_dir = augmentation_ori_dir, save_format = 'jpeg')
gen2 = datagen.flow(wound.truth_imgs, batch_size = 1, seed = seed, shuffle = False, save_to_dir = augmentation_truth_dir, save_format = 'jpeg')

i = 0

while True:
    x1 = gen1.next()
    x2 = gen2.next()

    i += 1
    if i >= total_count:
        break