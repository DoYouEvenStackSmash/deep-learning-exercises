#!/usr/bin/python3
from cgitb import text
from distutils.command.build import build
from operator import concat
import tensorflow as tf 

# Download and import the MIT 6.S191 package
import mitdeeplearning as mdl

# Import all remaining packages
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
from tensorflow.keras.layers import Dense
from batch_functions import *

def test_prediction(model, vectorized_songs, c2i_d = None, i2c_a = None):
  x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
  pred = model(x)
  print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
  print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

def untrained_prediction(model, vectorized_songs, c2i_d = None, i2c_a = None):
  x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
  pred = model(x)
  print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
  print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

  # to get preditions from the model, we sample from output distribution. 
  # This means we are using a categorical distribution to sample over the predition.
  # this gives a predition of the next character's index at each timestep.

  sampled_indices = tf.random.categorical(pred[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
  
  print("Input: \n", str(repr("".join(i2c_a[x[0]]))))
  print()
  print("Next Char Predictions: \n", repr("".join(i2c_a[sampled_indices])))
