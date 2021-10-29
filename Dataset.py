import pdb
import numpy as np
import tensorflow as tf
import random
import cv2
import os

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
#from tensorflow.contrib.data import Iterator
Iterator = tf.data.Iterator
import matplotlib.pyplot as plt
from utils import read_pgm_file


class WatermarkDataLoader(object):

    def __init__(self, main_dir, sub_dir, batch_size, resize_shape, crop_shape, paths_file, buffer_size=100, split='train'):
        self.main_dir = main_dir
        self.sub_dir = sub_dir
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.buffer_size = buffer_size
        self.paths_file = paths_file

        self.imgs_files = []
        self.labels_files = []

        # Read image and label paths from file and fill in self.images,
        # self.labels
        self.parse_file(self.paths_file)

        if split == 'train':
            self.shuffle_lists()
#         else:
#             half_size = len(self.imgs_files) // 2
#             self.imgs_files = self.imgs_files[:half_size]
#             self.labels_files = self.labels_files[half_size:half_size * 2]

        self.data_len = len(self.imgs_files)
        print('num of train: %d  num of valid: %d' %
              (len(self.imgs_files), len(self.labels_files)))

        img = convert_to_tensor(self.imgs_files, dtype=dtypes.string)
        label = convert_to_tensor(self.labels_files, dtype=dtypes.string)
        data_tr = tf.data.Dataset.from_tensor_slices((img, label))

        if split == 'train':
            # , num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.map(self.parse_train, num_parallel_calls=8)
            data_tr = data_tr.shuffle(buffer_size)
        else:
            # , num_threads=8, output_buffer_size=100*self.batch_size)
            data_tr = data_tr.map(self.parse_val, num_parallel_calls=8)

        data_tr = data_tr.batch(batch_size)
        self.data_tr = data_tr.prefetch(buffer_size=self.batch_size)

    def shuffle_lists(self):
        random.shuffle(self.imgs_files)
        random.shuffle(self.labels_files)

    def parse_train(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_jpeg(label, channels=0)

        # label= tf.concat([label, label, label], axis=-1)
#
#        # flipping
#        combined= tf.image.random_flip_left_right(combined)
#
#        # cropping
        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = tf.random_crop(
                img, [self.crop_shape[0], self.crop_shape[1], 3])
        else:
            img_crop = tf.image.resize_images(
                img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = tf.random_crop(
                label, [self.crop_shape[0], self.crop_shape[1], 1])
        else:
            label_crop = tf.image.resize_images(
                label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop > 0, tf.float32)

        return img_crop, label_crop

    def parse_val(self, im_path, label_path):
        # Load image
        img = tf.read_file(im_path)
        img = tf.image.decode_jpeg(img, channels=3)

        # Load label
        label = tf.read_file(label_path)
        label = tf.image.decode_jpeg(label, channels=1)

        # label= tf.concat([label, label, label], axis=-1)

        if img.shape[0] >= self.crop_shape[0] and img.shape[1] >= self.crop_shape[1]:
            img_crop = img[:self.crop_shape, :self.crop_shape, :]
        else:
            img_crop = tf.image.resize_images(
                img, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if label.shape[0] >= self.crop_shape[0] and label.shape[1] >= self.crop_shape[1]:
            label_crop = label[:self.crop_shape, :self.crop_shape, :]
        else:
            label_crop = tf.image.resize_images(
                label, self.resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img_crop = tf.cast(img_crop, tf.float32) / 255.
        label_crop = tf.cast(label_crop > 0, tf.float32)

        return img_crop, label_crop

    def parse_file(self, path):
        ff = open(path, 'r')
        for line in ff:
            tokens = line.strip().split(' ')
            self.imgs_files.append(self.main_dir + tokens[0])
            self.labels_files.append(
                self.sub_dir + tokens[0].replace('val', 'test'))

    def print_files(self):
        for x, y in zip(self.imgs_files, self.labels_files):
            print(x, y)
