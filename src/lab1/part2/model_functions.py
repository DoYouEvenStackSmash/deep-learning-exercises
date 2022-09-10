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
from input_processing import *
from tensorflow.keras.layers import Dense
from batch_functions import *
from vectorize_functions import *
from model_predict_examples import *
from batch_processing import *
from assertion_tests import *
from param_bank import *

# default_params = {
#   ### Hyperparameter setting and optimization ###
#   # Optimization parameters:
#   "num_training_iterations" : 2000,  # Increase this to train longer
#   "batch_size" : 4,  # Experiment between 1 and 64
#   "seq_length" : 100,  # Experiment between 50 and 500
#   "learning_rate" : 5e-3,  # Experiment between 1e-5 and 1e-1


# # Model parameters: 
#   "vocab_size" : 0,
#   "embedding_dim" : 256,
#   "rnn_units" : 1024  # Experiment between 1 and 2048
# }
# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    LSTM(rnn_units),
    Dense(vocab_size)
  ])
  return model

def build_model_wrapper(io, params = default_params):
  return build_model(len(io.i2c), params['embedding_dim'], params['rnn_units'], params['batch_size'])

'''labels = ground truth
  logits = predictions'''
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  return loss

@tf.function
def train_step(x, y, model, optimizer):
  with tf.GradientTape() as tape:
    y_hat = model(x)
    loss = compute_loss(y, y_hat)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss


def trainer(io, model, optimizer, params = default_params):
  ##################
  # Begin training!#
  ##################
  num_training_iterations = params['num_training_iterations']
  seq_length = params['seq_length']
  batch_size = params['batch_size']
  history = []
  plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
  if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

  for iter in tqdm(range(num_training_iterations)):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_batch(io.V, seq_length, batch_size)
    loss = train_step(x_batch, y_batch, model, optimizer)

    # Update the progress bar
    history.append(loss.numpy().mean())
    plotter.plot(history)

    # Update the model with the changed weights!
    if iter % 100 == 0:     
      model.save_weights(checkpoint_prefix)
      
  # Save the trained model and the weights
  model.save_weights(checkpoint_prefix)


# def main():
#   songs = mdl.lab1.load_training_data()
#   io = InputObj(songs)
#   io.process_strings()
#   is_vectorized_numpy(io)
#   batch_testing(io)
#   model = build_model_wrapper(io)
#   model.summary()
#   # test_prediction(model, io)

#   # process_strings(songs)

# main()