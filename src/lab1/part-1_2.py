#!/usr/bin/python3
from calendar import c
import tensorflow as tf

import mitdeeplearning as mdl

import numpy as np

import matplotlib.pyplot as plt

'''
  Computations on Tensors
'''


'''
  Create a computation graph consisting of TensorFlow operations
  Output a Tensor with value 76
'''
def add_two_tensors():
  # create nodes in the graph, and init values
  a = tf.constant(15)
  b = tf.constant(61)
  
  # add the values
  c1 = tf.add(a,b)
  c2 = a + b  # Note: TensorFlow overrides "+" to act on Tensors

  print(c1)
  print(c2)

def more_complicated_computation():
  # a = tf.constant(13)
  # b = tf.constant(17)
  a,b = 1.5,2.5
  ans = func(a,b)
  print(ans)

def func(a, b):
  a = tf.constant(a)
  b = tf.constant(b)
  c = tf.add(a, b) # 4
  d = tf.subtract(b, tf.constant(1.0)) # 1.5
  e = tf.multiply(c, d) # 4 * 1.5 = 6
  return e

def examples():
  add_two_tensors()

def exercises():
  more_complicated_computation()

def main():
  exercises()

main()

