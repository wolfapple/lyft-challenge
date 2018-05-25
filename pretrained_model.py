from __future__ import print_function
import argparse
import sys
import os
import glob
import shutil
import numpy as np
import tensorflow as tf
from keras.utils.data_utils import get_file
from tqdm import tqdm
from model import Deeplabv3

CKPT_DIR = "checkpoints"
WEIGHTS_DIR = "weights"

def main():
  # argument
  parser = argparse.ArgumentParser()
  parser.add_argument("-dataset", choices=["pascal", "cityscapes", "ade20k"], help="choose dataset", default="pascal")
  parser.add_argument("-backbone", choices=["mobilenetv2", "xception"], help="choose backbone", default="mobilenetv2")
  args = parser.parse_args()

  # set url
  if args.dataset == "pascal" and args.backbone == "mobilenetv2":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz"
  elif args.dataset == "pascal" and args.backbone == "xception":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz"
  elif args.dataset == "cityscapes" and args.backbone == "mobilenetv2":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz"
  elif args.dataset == "cityscapes" and args.backbone == "xception":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz"
  elif args.dataset == "ade20k" and args.backbone == "xception":
    ckpt_url = "http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_14.tar.gz"
  else:
    print("The model has only been pretrained using Xception_65")
    sys.exit()

  # download pretrained model
  ckpt_file = download(ckpt_url)

  # extract tensors from checkpoint
  extract_tensors(ckpt_file, args.backbone)

  # load weights and save
  save_weights(args.dataset, args.backbone)

def download(url):
  if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)
  ckpt = get_file(url.split("/")[-1], url, extract=True, cache_dir=CKPT_DIR, cache_subdir='')
  ckpt_dir = ckpt[:-18]
  ckpt_file = glob.glob(f'{ckpt_dir}/model.ckpt*.index')[0][:-6]
  return ckpt_file

def extract_tensors(ckpt_file, backbone):  
  if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)
  
  reader = tf.train.NewCheckpointReader(ckpt_file)
  for key in reader.get_variable_to_shape_map():
    if backbone == 'xception':
      filename = get_xception_filename(key)
    elif backbone == 'mobilenetv2':
      filename = get_mobilenetv2_filename(key)
    if filename:
      path = os.path.join(WEIGHTS_DIR, filename)
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

def save_weights(dataset, backbone):
  input_shapes = {'pascal':(512, 512, 3), 'cityscapes':(512, 512, 3), 'ade20k':(512, 512, 3)}
  num_classes = {'pascal':21, 'cityscapes':19, 'ade20k':151}
  model = Deeplabv3(weights=None, input_shape=(512, 512, 3), classes=num_classes[dataset], backbone=backbone)
  print("Loading weights...")
  for layer in tqdm(model.layers):
    if layer.weights:
      weights = []
      for weight in layer.weights:
        weight_name = os.path.basename(weight.name).replace(':0', '')
        weight_file = layer.name + '_' + weight_name + '.npy'
        weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
        weights.append(weight_arr)
      layer.set_weights(weights)

  print("Saving model weights...")
  model.save_weights("weights.h5")
  print("pretrained.h5 file created!")
  print("clear temp file...")
  shutil.rmtree(WEIGHTS_DIR)
  shutil.rmtree(CKPT_DIR)

if __name__=="__main__":
  main()