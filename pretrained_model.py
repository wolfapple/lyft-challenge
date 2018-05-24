from __future__ import print_function
import argparse
import sys
import os
import glob
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file

def main():
  # argument
  parser = argparse.ArgumentParser()
  parser.add_argument("-dataset", choices=["pascal_voc", "cityscapes", "ade20k"], help="choose dataset", default="pascal_voc")
  parser.add_argument("-backbone", choices=["mobilenet_v2", "xception_65"], help="choose backbone", default="mobilenet_v2")
  args = parser.parse_args()

  # set url
  if args.dataset == "pascal_voc" and args.backbone == "mobilenet_v2":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz"
  elif args.dataset == "pascal_voc" and args.backbone == "xception_65":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz"
  elif args.dataset == "cityscapes" and args.backbone == "mobilenet_v2":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz"
  elif args.dataset == "cityscapes" and args.backbone == "xception_65":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz"
  elif args.dataset == "ade20k" and args.backbone == "xception_65":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_14.tar.gz"
  else:
    print("The model has only been pretrained using Xception_65")
    sys.exit()

  # download pretrained model
  ckpt_file = download(ckpt_url)

  # extract tensors from checkpoint
  extract_tensors(ckpt_file, args.backbone)

def download(url):
  model_dir = "checkpoints"
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  ckpt = get_file(url.split("/")[-1], url, extract=True, cache_dir=model_dir, cache_subdir='')
  ckpt_dir = ckpt[:-18]
  ckpt_file = glob.glob(f'{ckpt_dir}/model.ckpt*.index')[0][:-6]
  return ckpt_file

def extract_tensors(ckpt_file, backbone):
  weight_dir = "weights"
  if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
  
  reader = tf.train.NewCheckpointReader(ckpt_file)
  for key in reader.get_variable_to_shape_map():
    if backbone == 'xception_65':
      filename = get_xception_filename(key)
    elif backbone == 'mobilenet_v2':
      filename = get_mobilenetv2_filename(key)
    if filename:
      path = os.path.join(weight_dir, filename)
      arr = reader.get_tensor(key)
      np.save(path, arr)

def get_xception_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('xception_65_', '')
    filename = filename.replace('decoder_', '', 1)
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
      return None
    if 'entry_flow' in filename or 'exit_flow' in filename:
      filename = filename.replace('_unit_1_xception_module', '')
    elif 'middle_flow' in filename:
      filename = filename.replace('_block1', '')
      filename = filename.replace('_xception_module', '')

    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def get_mobilenetv2_filename(key):
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('MobilenetV2_', '')
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
      return None

    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'

if __name__=="__main__":
  main()