#!/usr/bin/python3

# Download and import the MIT 6.S191 package
import mitdeeplearning as mdl

import numpy as np
# test example batch function
def batch_testing(vectorized_songs):
  # Perform some simple tests to make sure your batch function is working properly! 
  test_args = (vectorized_songs, 10, 2)
  if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
    not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
    not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
    print("======\n[FAIL] could not pass tests")
  else: 
    print("======\n[PASS] passed all tests!")


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
