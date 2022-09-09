#!/usr/bin/python3

from cgitb import text
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

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU

# assert len(tf.config.list_physical_devices('GPU')) > 0



def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

'''TODO: Add LSTM and Dense layers to define the RNN model using the Sequential API.'''
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #   of a fixed embedding size
    #input dim = vocab_size
    # embedding_dim = embedding_dim = output dim
    
    #  Input layer, trainable lookup table that maps the numbers of each character to a 
    #  vector with embedding_dim dimensions. cat_one representation
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    # TODO: Call the LSTM function defined above to add this layer.
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size. 
    # TODO: Add the Dense layer.
    # '''TODO: DENSE LAYER HERE'''
    # this transforms some input Tensor into a 1-D tensor
    Dense(vocab_size)
  ])

  return model

def sample_model_build(vocab):
  model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
  model.summary()
  return model
'''
# Build a simple model with default hyperparameters. You will get the 
#   chance to change these later.
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
'''
def text_vectorize_prep(vocab):
  ### Define numerical representation of text ###

  # Create a mapping from character to unique index.
  # For example, to get the index of the character "d", 
  #   we can evaluate `char2idx["d"]`.  
  char2idx = {u:i for i, u in enumerate(vocab)}

  # Create a mapping from indices to characters. This is
  #   the inverse of char2idx and allows us to convert back
  #   from unique index to the character in our vocabulary.
  idx2char = np.array(vocab)
  ''' now we have our pair mapping
  char2idx: char -> index in idx2char
  idx2char: index -> char at idx2char index
  '''
  return char2idx, idx2char

  # print('{')
  # for char,_ in zip(char2idx, range(20)):
  #   print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
  # print('  ...\n}')
  
def vectorize_string(all_songs_string, char2idx_dict, idx2char_arr = None):
  np_str_arr = []
  for i in range(len(all_songs_string)):
    np_str_arr.append(char2idx_dict[all_songs_string[i]])
  return np.array(np_str_arr)

def vectorize_songs(concat_songs):
  unique_notes = sorted(set(concat_songs))
  c2i_dict,i2c_arr = text_vectorize_prep(unique_notes)
  vectorized_songs = vectorize_string(concat_songs, c2i_dict)
  assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"
  return vectorized_songs, c2i_dict, i2c_arr

'''
  vectorized_songs is a concatenated string of integers, as a numpy array
  seq_length is an integer defining the length of the sequences
  batch_size is the number of sequences to choose in the batch
'''
def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  # vectorized songs is a 1 dimensional np.array of n elements
  
  n = vectorized_songs.shape[0] - 1 # subtract 1 to prevent out of bounds in output batch
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  '''TODO: construct a list of input sequences for the training batch'''
  '''
  As input, we want the sequences of seq_length starting at each index in idx
  seqs = []
  for i in idx:
    seqs.append(vectorized_songs[i : i + seq_length])
  seqs = [vectorized_songs[i : i + seq_length] for i in idx]
  '''
  
  input_batch = [vectorized_songs[i : i + seq_length] for i in idx]
  '''TODO: construct a list of output sequences for the training batch'''
  '''
  As output, we want the sequences of input, shifted to the right by 1.  Can this go out of bounds? no. as per n above
  seqs = []
  for i in idx:
    seqs.append(vectorized_songs[i + 1 : i + seq_length + 1])
  seqs = [vectorized_songs[i + 1 : i + seq_length + 1] for i in idx]
  '''
  output_batch = [vectorized_songs[i + 1 : i + seq_length + 1] for i in idx]

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

def batch_testing(vectorized_songs):
  # Perform some simple tests to make sure your batch function is working properly! 
  test_args = (vectorized_songs, 10, 2)
  if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
    not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
    not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
    print("======\n[FAIL] could not pass tests")
  else: 
    print("======\n[PASS] passed all tests!")

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

  sampled_indices = tf.random.categorical(pred[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
  
  print("Input: \n", str(repr("".join(i2c_a[x[0]]))))
  print()
  print("Next Char Predictions: \n", repr("".join(i2c_a[sampled_indices])))



def main():
  songs = mdl.lab1.load_training_data()
  # Join our list of song strings into a single string containing all songs
  songs_joined = "\n\n".join(songs) 
  vs, c2i_d, i2c_a = vectorize_songs(songs_joined)
  m = sample_model_build(i2c_a)
  # test_prediction(m, vs, c2i_d,i2c_a)
  untrained_prediction(m, vs, c2i_d,i2c_a)
  # batch_testing(vs)
  return

  # Find all unique characters in the joined string
  vocab = sorted(set(songs_joined))
  text_vectorize_prep(vocab)
  # print("There are", len(vocab), "unique characters in the dataset")
  return

# Print one of the songs to inspect it in greater detail!
  example_song = songs[0]
  print("\nExample song: ")
  print(example_song)
  print("hello test")

main()