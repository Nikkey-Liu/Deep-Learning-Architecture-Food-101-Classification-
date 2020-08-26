import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict
import cv2
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
with open('food-101/meta/train.txt', 'r') as txt:
    train_img_Name = [l.strip() for l in txt.readlines()]#读取每一行数据
with open('food-101/meta/test.txt', 'r') as txt:
    test_img_Name = [l.strip() for l in txt.readlines()]#读取每一行数据
with open('food-101/meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]#读取每一行数据
    class_to_ix = dict(zip(classes, range(len(classes))))#将数据转化为0，1，2，3，4...的index类型
    ix_to_class = dict(zip(range(len(classes)), classes))#将数据从index 转化为classname
    class_to_ix = {v: k for k, v in ix_to_class.items()}#dic类型数据保存
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))#orderde by input
print(sorted_class_to_ix)#检查输出结果

import glob

pthroot='food-101/images/'
#读取数据在 classes txt 文本中
test_img_datas=[]
test_img_labs=[]
for imgname in test_img_Name:
    imgpath = pthroot+imgname+".jpg"
    test_img_datas +=glob.glob(imgpath)
for imgname in test_img_Name:
    subdir = imgname.split("/")[0]
    class_ix = class_to_ix[subdir]
    test_img_labs.append(class_ix)
train_img_datas=[]
train_img_labs=[]
for imgname in train_img_Name :
    imgpath = pthroot+imgname+".jpg"
    train_img_datas+= glob.glob(imgpath)
for imgname in train_img_Name:
    subdir = imgname.split("/")[0]
    class_ix = class_to_ix[subdir]
    train_img_labs.append(class_ix)
print(test_img_labs.__len__())
print(test_img_datas.__len__())
print(train_img_labs.__len__())
print(train_img_datas.__len__())
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
index=[i for i in range(train_img_datas.__len__())]
np.random.shuffle(index)
tfrecord_filename = 'food-101/train.tfrecord'
writer = tf.compat.v1.python_io.TFRecordWriter(tfrecord_filename)
for i in range(train_img_datas.__len__()):
    im_d=train_img_datas[index[i]]
    im_l=train_img_labs[index[i]]
    data=cv2.imread(im_d,1)
    image_shape = data.shape
    image_string = open(im_d, 'rb').read()
    example = tf.train.Example(
        features= tf.train.Features(
            feature={
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'depth': _int64_feature(image_shape[2]),
                "image": _bytes_feature(image_string),
                "label": _int64_feature(im_l),
            }
        )
    )
    writer.write(example.SerializeToString())
writer.close()

