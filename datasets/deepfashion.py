# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the DeepFashion dataset.

The dataset scripts used to create the dataset is modified from:
tensorflow/models/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils
import pickle
import pdb

slim = tf.contrib.slim

_FILE_PATTERN = '%s_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': None, 'test': None}

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image_raw_0': 'A color image of varying size.',
    'image_raw_1': 'A color image of varying size.',
    'label': 'A single integer between 0 and 1',
    'id_0': 'A single integer',
    'id_1': 'A single integer',
}


from tensorflow.python.ops import parsing_ops
def get_split(split_name, dataset_dir, data_name='DeepFashion', file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading DeepFashion.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % (data_name, split_name))

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader


  keys_to_features = {
     'image_raw_0' : tf.FixedLenFeature([], tf.string),
     'image_raw_1' : tf.FixedLenFeature([], tf.string),
     'label': tf.FixedLenFeature([], tf.int64), # For FixedLenFeature, [] means scalar
     'id_0': tf.FixedLenFeature([], tf.int64),
     'id_1': tf.FixedLenFeature([], tf.int64),
     'cam_0': tf.FixedLenFeature([], tf.int64),
     'cam_1': tf.FixedLenFeature([], tf.int64),
     'image_format': tf.FixedLenFeature([], tf.string, default_value='jpg'),
     'image_height': tf.FixedLenFeature([], tf.int64, default_value=256),
     'image_width': tf.FixedLenFeature([], tf.int64, default_value=256),
     'real_data': tf.FixedLenFeature([], tf.int64, default_value=1),
     'pose_peaks_0': tf.FixedLenFeature([16*16*18], tf.float32),
     'pose_peaks_1': tf.FixedLenFeature([16*16*18], tf.float32),
     'pose_mask_r4_0': tf.FixedLenFeature([256*256*1], tf.int64),
     'pose_mask_r4_1': tf.FixedLenFeature([256*256*1], tf.int64),
     
     'shape': tf.FixedLenFeature([1], tf.int64),
      'indices_r4_0': tf.VarLenFeature(dtype=tf.int64),
      'values_r4_0': tf.VarLenFeature(dtype=tf.float32),
      'indices_r4_1': tf.VarLenFeature(dtype=tf.int64),
      'values_r4_1': tf.VarLenFeature(dtype=tf.float32),
     'pose_subs_0': tf.FixedLenFeature([20], tf.float32),
     'pose_subs_1': tf.FixedLenFeature([20], tf.float32),
  }

  items_to_handlers = {
      'image_raw_0': slim.tfexample_decoder.Image(image_key='image_raw_0', format_key='image_format'),
      'image_raw_1': slim.tfexample_decoder.Image(image_key='image_raw_1', format_key='image_format'),
      'label': slim.tfexample_decoder.Tensor('label'),
      'id_0': slim.tfexample_decoder.Tensor('id_0'),
      'id_1': slim.tfexample_decoder.Tensor('id_1'),
      'pose_peaks_0': slim.tfexample_decoder.Tensor('pose_peaks_0',shape=[16*16*18]),
      'pose_peaks_1': slim.tfexample_decoder.Tensor('pose_peaks_1',shape=[16*16*18]),
      'pose_mask_r4_0': slim.tfexample_decoder.Tensor('pose_mask_r4_0',shape=[256*256*1]),
      'pose_mask_r4_1': slim.tfexample_decoder.Tensor('pose_mask_r4_1',shape=[256*256*1]),

      'pose_sparse_r4_0': slim.tfexample_decoder.SparseTensor(indices_key='indices_r4_0', values_key='values_r4_0', shape_key='shape', densify=False),
      'pose_sparse_r4_1': slim.tfexample_decoder.SparseTensor(indices_key='indices_r4_1', values_key='values_r4_1', shape_key='shape', densify=False),
      
      'pose_subs_0': slim.tfexample_decoder.Tensor('pose_subs_0',shape=[20]),
      'pose_subs_1': slim.tfexample_decoder.Tensor('pose_subs_1',shape=[20]),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  print('load pn_pairs_num ......')
  fpath = os.path.join(dataset_dir, 'pn_pairs_num_'+split_name+'.p')
  with open(fpath,'r') as f:
    pn_pairs_num = pickle.load(f)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=pn_pairs_num,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)



