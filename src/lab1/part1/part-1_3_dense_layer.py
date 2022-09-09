#!/usr/bin/python3
import tensorflow as tf

import mitdeeplearning as mdl

import numpy as np

import matplotlib.pyplot as plt
# import keras.api._v2.keras as keras
from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


def model_summary(model):
  # note: parameters = total number of weights + total number of bias

  model.summary()
'''
  sequential model from keras and a single Dense layer to define our network.
'''
def dense_mode_thing():
  n_output_nodes = 3
  model = Sequential() # initialize sequential model
  
  
  # Note: model is not yet built, cannot call summary
  # print("\n\npre_layer")
  # model_summary(model)
  

  # initialized dense layer with 3 units
  dense_layer = Dense(n_output_nodes, activation='sigmoid')
  model.add(dense_layer)

  # Note: model is not yet built, cannot call summary
  # print("\n\npost_layer")
  # model_summary(model)
  # print(model.output_shape())
  x_input = tf.constant([[1,2.]], shape=(1,2))
  model_output = model(x_input)
  print("\n\npost call")
  print(model_output.numpy())
  '''
    for a dense layer with 3 units:
      each input gets a weight.
        z1 = i1 * w1
  '''
  model_summary(model)
  

'''
  Model class groups layers together for model training and inference.
  Forward pass through the network is defined using the call function.

  Allows us to define custom layers, custom training loops, custom 
  activation functions, and custom models.

  This is an alternative to the sequential model.
'''
class SubclassModel(tf.keras.Model):
  # in __init__ we define the Model's layers
  def __init__(self, n_output_nodes):
    super(SubclassModel, self).__init__()
    '''TODO: Our model consists of a single Dense Layer.'''
    self.dense_layer = Dense(n_output_nodes, activation="sigmoid")

  # in the call function, we define the model's forward pass
  def call(self, inputs):
    x = self.dense_layer(inputs)
    print("shape of dense layer output: {}".format(tf.shape(x).numpy()))
    return x

def subclass_test():
  n_output_nodes = 3
  model = SubclassModel(n_output_nodes) # initialize model
  
  x_input = tf.constant([[1,2.]], shape=(1,2))
  model_output = model.call(x_input) # call on input
  
  print(model_output.numpy())
  # model_summary(model)
  # mdl.lab1.test_custom_dense_layer_output(model_output)

class IdentityModel(tf.keras.Model):
  # in __init__ we define our model's layers
  def __init__(self, n_output_nodes):
    super(IdentityModel, self).__init__()
    self.dense_layer = Dense(n_output_nodes, activation="sigmoid")
  
  #add a flag to output input, unchaged, under control of isidentity
  def call(self, inputs, isidentity=False):
    if isidentity:
      return inputs
    x = self.dense_layer(inputs)
    return x

def identity_test():
  n_output_nodes = 3
  model = IdentityModel(n_output_nodes)
  x_input = tf.constant([[1,2.]], shape=(1,2))
  out_activate = model.call(x_input) # call normal
  out_identity = model.call(x_input, True) # call with identity flag
  print("Network output with activation: {};\nnetwork identity output: {}".format(out_activate.numpy(), out_identity.numpy()))

def main():
  identity_test()
  # subclass_test()
  # dense_mode_thing()

main()