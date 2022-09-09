#!/usr/bin/python3

import tensorflow as tf

import mitdeeplearning as mdl

import numpy as np

import matplotlib.pyplot as plt

### defining a network layer

# n_output_nodes: number of output nodes
# input shape: shape of the input
# x: input to the layer

class OurDenseLayer(tf.keras.layers.Layer):
  def __init__(self, n_output_nodes):
    super(OurDenseLayer, self).__init__()
    self.n_output_nodes = n_output_nodes

  def build(self, input_shape):
    d = int(input_shape[-1])
    # print(f"dimensionality of weights{d}")
    
    # Define and initialize parameters: weight matrix W and bias b
    # note that parameter initialization is random!
    
    # weight is dimensionality according to input shape
    # by definition this is a transpose
    self.W = self.add_weight("weight", shape=[d, self.n_output_nodes]) # note the dimensionality
    
    # bias is a...1d vector
    self.b = self.add_weight("bias", shape=[1, self.n_output_nodes]) # note the dimensionality

  def call(self, x):
    '''TODO: define the operation for z (hint: use tf.matmul'''
    # multiply the tensor x by the weights, and add bias
    #print(self.W.numpy())
    z = tf.matmul(x,self.W) + self.b
    #print("`z` is a {}-d Tensor with shape: {}".format(tf.rank(z).numpy(), tf.shape(z)))

    '''TODO: define the operation for out (hint: use tf.sigmoid)'''
    # apply nonlinearity
    y = tf.math.sigmoid(z)
    return y
  
def layer_params():
  tf.random.set_seed(1)
  layer = OurDenseLayer(3)
  layer.build((1,2)) # 2 elements in 1-d Tensor
  # 2. is 2.0
  # doesnt seem to matter whether shape = tuple or list
  x_input = tf.constant([[1,2.]], shape=(1,2))
  # print("`x_input` is a {}-d Tensor with shape: {}".format(tf.rank(x_input).numpy(), tf.shape(x_input)))
  # print(x_input.numpy())
  # return
  # print("call!\n")
  y = layer.call(x_input)
  print(y.numpy())
  mdl.lab1.test_custom_dense_layer_output(y)

def main():
  layer_params()

main()