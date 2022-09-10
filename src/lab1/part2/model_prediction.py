#!/usr/bin/python3
from distutils.command.build import build
from json import load
from operator import concat
import wave
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
from model_trainer import loading_wrapper


### Prediction of a generated song ###
def generate_text(model, io, start_string, generation_length=1000):
  # Evaluation step (generating ABC text using the learned RNN model)

  '''TODO: convert the start string to numbers (vectorize)'''
  # io = InputObj(start_string)
  # io.process_strings()
  input_eval = [io.c2i[start_string]]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      '''TODO: evaluate the inputs and generate the next character predictions'''
      predictions = model(input_eval)
      
      # Remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      
      '''TODO: use a multinomial distribution to sample'''
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      
      # Pass the prediction along with the previous hidden state
      #   as the next inputs to the model
      input_eval = tf.expand_dims([predicted_id], 0)
      
      '''TODO: add the predicted character to the generated text!'''
      # Hint: consider what format the prediction is in vs. the output
      text_generated.append(io.i2c[predicted_id])
    
  return (start_string + ''.join(text_generated))


def make_song():
  songs = mdl.lab1.load_training_data()
  io = InputObj(songs)
  io.process_strings()
  model = loading_wrapper(io)
  model.summary()
  song = generate_text(model, io, start_string="X", generation_length=1000)
  # return song
  print(song)
  generated_songs = mdl.lab1.extract_song_snippet(song)
  for i, s in enumerate(generated_songs):
    waveform = mdl.lab1.play_song(s)
    if waveform:
      print("generated song", i)
      ipythondisplay.display(waveform)

def main():
  make_song()
  return
  
  songs = mdl.lab1.load_training_data()
  io = InputObj(songs)
  io.process_strings()
  model = loading_wrapper(io)
  model.summary()
  song = generate_text(model, io, start_string="x")
  print(song)

main()
