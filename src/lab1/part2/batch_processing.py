#!/usr/bin/python3

import numpy as np

def batch_wrapper(io, seq_len, batch_size):
  return get_batch(io.V, seq_len, batch_size)

def get_batch(V, seq_len, batch_size):
  V_len = V.shape[0] - 1 # length of string
  r_idx = np.random.choice(V_len - seq_len, batch_size)

  input_batch = [V[i : i + seq_len] for i in r_idx]
  target_batch = [V[i + 1 : i + seq_len + 1] for i in r_idx]
  
  # shift from 1d vector to 2d matrix where each row is seq
  x_batch = np.reshape(input_batch, [batch_size, seq_len])
  y_batch = np.reshape(target_batch, [batch_size, seq_len])

  return x_batch, y_batch
