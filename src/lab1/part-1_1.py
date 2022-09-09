#!/usr/bin/python3

'''
  Introduction to Tensorflow
    Tensoflow is named as such because it handles the flow of Tensors, 
    n-dimensional arrays of base datatypes such as strings/integers.
    
    Tensors provide a way to generalize vectors and matrices to higher dimensions.
'''

import tensorflow as tf

import mitdeeplearning as mdl

import numpy as np

import matplotlib.pyplot as plt

'''
  Shape: 
    Attribute of a Tensor defining the number of dimensions and 
    the size of each dimension.

  Rank: 
    Attribute of a Tensor defining the number of dimensions;
    Can also think of this as its Order or Degree
'''

# 0-d Tensor, of which a scalar is an example

def zero_dimensional_tensor():
  # tf.constant name comes from being created as a Const node in the tensor graph.
  sport = tf.constant("Tennis", tf.string)
  number = tf.constant(1.41421356237, tf.float64)
  print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
  print("`number` is a {}-d Tensor".format(tf.rank(number.numpy())))

def one_dimensional_tensor():
  sports = tf.constant(["Tennis", "Basketball"], tf.string)
  numbers = tf.constant([3.5141592, 1.414213, 2.71821], tf.float64)
  print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
  print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))


def two_dimensional_tensor():
  #matrix = #todo
  matrix = tf.constant([["image height"], ["image width"]], tf.string)
  print("`matrix` is a {}-d Tensor with shape: {}".format(tf.rank(matrix).numpy(), tf.shape(matrix)))
  # `matrix` is a 2-d Tensor with shape: [2 1] 2 dimensions of size 1 each
  assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
  assert tf.rank(matrix).numpy() == 2

'''
  here the dimensions correspond to the number of example images in our batch, 
  image height, image width, and the number of color channels

  Use tf.zeroes to initialize a 4-d Tensor with zeros with size 10 x 256 x 256 x 3
  Think of this as:
    10 images
    256 height
    256 width
    3 color channels
'''

def four_dimensional_tensor():
  # 2d tensor 256x256
  #1d tensor of 3 elements
  # 10 elements each of an image where each pixel is 3 integers

  # colors = tf.constant([tf.zeros([1, 3])]
  # image_matrix = ([256, 3])
  # matrix = tf.zeros([2, 256])
  images = tf.zeros((10, 256, 256,3))
  print("`images` is a {}-d tensor with shape: {}".format(tf.rank(images).numpy(), tf.shape(images)))

  assert isinstance(images, tf.Tensor), "images must be a tf Tensor object"
  assert tf.rank(images).numpy() == 4, "images must be of rank 4"
  assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
  return images

def slice_tensor():
  # matrix = tf.constant([[1,2], [3,4]], tf.int32)
  matrix = tf.zeros((4, 6, 8 ,3))
  row_vector = matrix[1] # 6 of 8 of 3
  # column_vector = matrix[:,0] # contiguous slice of first dimension, specific element of second dimension
  column_vector = matrix[:,:, 7] # contiguous slice of first dimension, contiguuous slice of second dimension, specific element of third dimension
  scalar = matrix[1,2]
  print("`row_vector`: {}".format(row_vector.numpy()))
  print("`column_vector` shape = {}:\n {}".format(tf.shape(column_vector).numpy(), column_vector.numpy()))
  print("`scalar`: {}".format(scalar.numpy()))

def examples():
  print(f"0-d tensor")
  zero_dimensional_tensor()
  print(f"1-d tensor")
  one_dimensional_tensor()

def exercises():
  print(f"2-d tensor")
  two_dimensional_tensor()
  print(f"4-d tensor")
  four_dimensional_tensor()

def main():
  # exercises()
  slice_tensor()

main()
