# v2 fixed two bugs and added kernel randomization to vary blob size/intensity
#  1) Overflow when adding original MNIST to blobs (if overlapping)
#  2) Test images classified as BLOBS were not guaranteed to have blobs

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy import signal
import scipy.ndimage.filters as fi
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
import gzip
import os
import tempfile

from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

""" 
CONFIGURATION PARAMETERS FOR NEW DATASET 
"""
p_blob = 0.5         # probability blobs get added to image
p_pixel = 0.0015     # probability of pixel (blob center) turning on
NORM_G = False       # normalize the blobs to a chosen peak intensity
blob_peak = 255      # peak intensity of blob (center pixel value)
MIN_KERNEL_SIZE = 5  # min gaussian kern size (gkern_size x gkern_size)
MAX_KERNEL_SIZE = 5 # max gaussian kernel size (gkern_size x gkern_size)
STD_MIN = np.sqrt(3) #0.1
STD_MAX = np.sqrt(3) # 2          # std dev of 2d gaussian filter
basesavedir = '/home/doleas/python/tensorflow/'
savedir = basesavedir + 'mnist_blobs_Ppixel0p0015_kernlen5_kernstd1p73_noNorm'

if not os.path.isdir(savedir):
  os.mkdir(savedir)
  os.mkdir(savedir + '/input_data')
  print('Made Directory: ' + savedir)

""" Method to quickly plot images: used for troubleshooting """
def plot_mnist(data, index=0):
  fig, ax = plt.subplots()
  ax.imshow(data[index, :, :, 0])
  plt.show()

""" Method to read in 32 bits at a time """
def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

"""
Read in the MNIST training images
"""
with gfile.Open('/home/doleas/python/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz', 'rb') as f:
   with gzip.GzipFile(fileobj=f) as bytestream:
     magic = _read32(bytestream)
     if magic != 2051:
       raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
     num_images = _read32(bytestream)
     rows = _read32(bytestream)
     cols = _read32(bytestream)
     buf = bytestream.read(rows * cols * num_images)
     data = np.frombuffer(buf, dtype=np.uint8)
     data = data.reshape(60000, 28, 28, 1)


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

"""
Read in the MNIST training labels
"""
#with gfile.Open('/home/doleas/python/tensorflow/mnist/input_data/train-labels-id#x1-ubyte.gz', 'rb') as f:
#  with gzip.GzipFile(fileobj=f) as bytestream:
#    magic = _read32(bytestream)
#    if magic != 2049:
#      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
#                       (magic, f.name))
#    num_items = _read32(bytestream)
#    buf = bytestream.read(num_items)
#    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
#    if one_hot:
#      return dense_to_one_hot(labels, num_classes)
#    return labels

"""
Generate and Add Blobs to MNIST dataset; create new labels
"""
""" Method to generate 2d gaussian kernel """
def gkern(kernlen=5, nsig=1):
  inp = np.zeros((kernlen, kernlen))
  inp[kernlen//2, kernlen//2]=1
  return fi.gaussian_filter(inp, nsig)

num_images=data.shape[0]
labels_blobs = np.random.binomial(1, p_blob, num_images).astype('uint8')
data_blobs = np.ndarray([num_images, 28, 28, 1], 'uint8')
for item in range(num_images):
  if labels_blobs[item]==1:
    blobs_base = (255*np.random.binomial(1, p_pixel, (28,28)))
    while np.sum(blobs_base) == 0: # repeat until at least 1 blob is added
      blobs_base = (255*np.random.binomial(1, p_pixel, (28,28)))
    gkern_size = np.random.random_integers(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE)
    std = STD_MIN + (STD_MAX-STD_MIN) * np.random.random_sample()
    blobs_gauss = signal.convolve2d(blobs_base, gkern(gkern_size, std), mode='same')
    if NORM_G:
      blobs_gauss_use = blobs_gauss * (blob_peak / max(1, blobs_gauss.max()))
      blobs_gauss_use = blobs_gauss_use.reshape([1, 28, 28, 1])
    else:
      blobs_gauss_use = blobs_gauss.reshape([1, 28, 28, 1])
  else:
    blobs_gauss_use = np.zeros([1, 28, 28, 1])
  orig_plus_blob = data[item, :, :, 0].astype('uint32') + blobs_gauss_use[0, :, :, 0].astype('uint32')
  orig_plus_blob = orig_plus_blob.clip(0, 255).astype('uint8')
  data_blobs[item, :, :, 0] = orig_plus_blob.reshape(28, 28)


"""
WRITE BLOB DATASET
"""
target_img_file = savedir + '/input_data/train-images-idx3-ubyte.gz'
#img_hdr = np.uint8([2051, 60000, 28, 28])
data_blobs_write = data_blobs.reshape(num_images * data.shape[1] * data.shape[2])

with gzip.open(target_img_file, 'wb') as f:
  #f.write(img_hdr)
  f.write(data_blobs_write)

target_label_file = savedir + '/input_data/train-labels-idx1-ubyte.gz'
#lbl_hdr = np.uint32([2049]).tobytes()
#labels_blobs_write = labels_blobs.tobytes()
with gzip.open(target_label_file, 'wb') as f:
  #f.write(lbl_hdr)
  f.write(labels_blobs)


  
"""
READ BACK BLOB DATASET
"""
"""
Read in the MNIST blob training images
"""
#with gfile.Open('/home/doleas/python/tensorflow/mnist_blobs/input_data/train-images-idx3-ubyte.gz', 'rb') as f:
with gfile.Open(savedir + '/input_data/train-images-idx3-ubyte.gz', 'rb') as f:  
   with gzip.GzipFile(fileobj=f) as bytestream:
     #magic = _read32(bytestream)
     #if magic != 2051:
     #  raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
     #num_images = _read32(bytestream)
     #rows = _read32(bytestream)
     #cols = _read32(bytestream)
     #print(str(magic) + " " + str(num_images) + " " + str(rows) + " " + str(cols))
     #buf = bytestream.read(rows * cols * num_images)
     buf = bytestream.read(28 * 28 * 60000)
     data_im = np.frombuffer(buf, dtype=np.uint8)
     #data = data.reshape(num_images, rows, cols, 1)
     data_im = data_im.reshape(60000, 28, 28, 1)

"""
Read in the MNIST blob training labels
"""
#with gfile.Open('/home/doleas/python/tensorflow/mnist_blobs/input_data/train-labels-idx1-ubyte.gz', 'rb') as f:
with gfile.Open(savedir + '/input_data/train-labels-idx1-ubyte.gz', 'rb') as f:
   with gzip.GzipFile(fileobj=f) as bytestream:
     #magic = _read32(bytestream)
     #if magic != 2051:
     #  raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
     #num_images = _read32(bytestream)
     #rows = _read32(bytestream)
     #cols = _read32(bytestream)
     #print(str(magic) + " " + str(num_images) + " " + str(rows) + " " + str(cols))
     #buf = bytestream.read(rows * cols * num_images)
     buf = bytestream.read(60000)
     data_ = np.frombuffer(buf, dtype=np.uint8)
     #data = data.reshape(num_images, rows, cols, 1)
     #data_ = data.reshape(60000, 28, 28, 1)


     
""" REDO ALL OF THE ABOVE FOR TEST DATA """
"""
Read in the MNIST training images
"""
with gfile.Open('/home/doleas/python/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
   with gzip.GzipFile(fileobj=f) as bytestream:
     magic = _read32(bytestream)
     if magic != 2051:
       raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
     num_images = _read32(bytestream)
     rows = _read32(bytestream)
     cols = _read32(bytestream)
     buf = bytestream.read(rows * cols * num_images)
     data = np.frombuffer(buf, dtype=np.uint8)
     data = data.reshape(num_images, rows, cols, 1)


"""
Generate and Add Blobs to MNIST dataset; create new labels
"""

num_images=data.shape[0]
labels_blobs = np.random.binomial(1, p_blob, num_images).astype('uint8')

data_blobs = np.ndarray([num_images, 28, 28, 1], 'uint8')
for item in range(num_images):
  if labels_blobs[item]==1:
    blobs_base = (255*np.random.binomial(1, p_pixel, (28,28)))
    while np.sum(blobs_base) == 0:
      blobs_base = (255*np.random.binomial(1, p_pixel, (28,28)))
    gkern_size = np.random.random_integers(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE)
    std = STD_MIN + (STD_MAX-STD_MIN) * np.random.random_sample()
    blobs_gauss = signal.convolve2d(blobs_base, gkern(gkern_size, std), mode='same')
    if NORM_G:
      blobs_gauss_use = blobs_gauss * (blob_peak / max(1, blobs_gauss.max()))
      blobs_gauss_use = blobs_gauss_use.reshape([1, 28, 28, 1])
    else:
      blobs_gauss_use = blobs_gauss.reshape([1, 28, 28, 1])
  else:
    blobs_gauss_use = np.zeros([1, 28, 28, 1])
  orig_plus_blob = data[item, :, :, 0].astype('uint32') + blobs_gauss_use[0, :, :, 0].astype('uint32')
  orig_plus_blob = orig_plus_blob.clip(0, 255).astype('uint8')
  data_blobs[item, :, :, 0] = orig_plus_blob.reshape(28, 28)


"""
WRITE BLOB DATASET
"""
target_img_file = savedir + '/input_data/t10k-images-idx3-ubyte.gz'
#img_hdr = np.uint8([2051, 60000, 28, 28])
data_blobs_write = data_blobs.reshape(num_images * data.shape[1] * data.shape[2])

with gzip.open(target_img_file, 'wb') as f:
  #f.write(img_hdr)
  f.write(data_blobs_write)

target_label_file = savedir + '/input_data/t10k-labels-idx1-ubyte.gz'
#lbl_hdr = np.uint32([2049]).tobytes()
#labels_blobs_write = labels_blobs.tobytes()
with gzip.open(target_label_file, 'wb') as f:
  #f.write(lbl_hdr)
  f.write(labels_blobs)
