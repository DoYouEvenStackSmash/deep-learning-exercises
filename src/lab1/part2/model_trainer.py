#!/usr/bin/python3
from cgitb import text
from distutils.command.build import build
from json import load
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
from batch_processing import *
from assertion_tests import *
from model_functions import *

'''
  Inference step with batch size of 1. Model will only be able to accept a fixed batch
  size once it is built.

  Rebuild model and restore weights from the latest checkpoint
'''
def loading_wrapper(io):
  model = build_model_wrapper(io, seed_params)
  model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  model.build(tf.TensorShape([1, None]))
  model.summary()
  return model
  
def training_wrapper(io):
  model = build_model_wrapper(io, mild_params)
  before_summary = f"{model.summary()}"
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
  trainer(io, model, optimizer, mild_params)
  after_summary = f"{model.summary()}"
  print(f"\n{'-' * 25}BEFORE{'-' * 25}\n{before_summary}")
  print(f"\n{'-' * 25}AFTER{'-' * 25}\n{after_summary}")

def main():
  songs = mdl.lab1.load_training_data()
  io = InputObj(songs)
  io.process_strings()
  loading_wrapper(io)
  # training_wrapper(io)

main()