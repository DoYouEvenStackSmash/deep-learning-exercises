#!/usr/bin/python3
import tensorflow as tf

import mitdeeplearning as mdl

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

### gradient computation

'''
  tf.GradientTape is used to trace operations for computing gradients.
  

  When a forward pass is made through the network, all 
  forward pass operations are recorded to a tape.  To compute
  the gradient, the tape is played bakcwards.
'''

def gradient_computation_example():
  x = tf.Variable(3.0)
  # initiate the gradient tape
  with tf.GradientTape() as tape:
    y = x * x
  
  # access the gradient -- derivative of y with respect to x
  
  # gradient is for tensors x and y
  dy_dx = tape.gradient(y, x)
  assert dy_dx.numpy() == 6.0
  print(dy_dx)

'''
  Differentiation and stochastic gradient descent are used to optimize
  a loss function.
'''
### Function minimization with automatic differentiation and SGD ###

'''
  Example: seeking L = (x - x_f) ^ 2'''
def optimize_loss_fxn():
  # initialize random value for initial x
  x = tf.Variable([tf.random.normal([1])])
  print("Initializing x={}".format(x.numpy()))
  # recall rank is the number of dimensions
  # number of elements in shape is the number of dimensions. each element is the size of the respective dimension
  print("`x` is a {}-d Tensor with shape: {}".format(tf.rank(x).numpy(), tf.shape(x)))

  learning_rate = 1e-2 # learning rate for SGD
  history = []
  x_f = 4 # target value

  '''
    run SGD for a number of iterations. 
    After each iteration:
      compute the loss
      compute the derivative of the loss with respect to x
      perform the SGD update
  '''
  num_iter = 500
  for i in range(num_iter):
    with tf.GradientTape() as tape:
      loss = (x - x_f) ** 2 # tf should override **
    grad = tape.gradient(loss, x) # compute derivative of loss with respect to x
    new_x = x - learning_rate * grad # sgd update
    x.assign(new_x) # update value of x
    history.append(x.numpy()[0]) # specific index  of 2d Tensor
  # plt.figure()
  a = [b for b in range(500)]
  plt.plot(history)
  plt.plot([0, num_iter], [x_f, x_f])
  plt.legend(('Predicted', 'True'))
  plt.xlabel('Iteration')
  plt.ylabel('x value')
  plt.show()
  # print(history)


  


def main():
  # matplotlib.use("svg")
  # matplotlib.use("agg")
  optimize_loss_fxn()
  # gradient_computation_example()

main()