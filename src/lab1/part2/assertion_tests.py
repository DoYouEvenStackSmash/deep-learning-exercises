#!/usr/bin/python3
from batch_processing import *
import mitdeeplearning as mdl
from input_processing import *
import tensorflow as tf 

def batch_testing(io):
  # Perform some simple tests to make sure your batch function is working properly! 
  test_args = (io.V, 10, 2)
  if not mdl.lab1.test_batch_func_types(get_batch, test_args) or \
    not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or \
    not mdl.lab1.test_batch_func_next_step(get_batch, test_args): 
    print("======\n[FAIL] could not pass tests")
  else: 
    print("======\n[PASS] passed all tests!")
  
def is_vectorized_numpy(io):
  assert isinstance(io.V, np.ndarray), "returned result should be a numpy array"

def test_prediction(model, io):
  x, y = get_batch(io.V, 100, batch_size=4)
  print("x shape")
  print(x.shape)
  pred = model(x)
  print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
  print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")

def untrained_prediction(model, io):
  x, y = get_batch(io.V, 100, batch_size=4)
  pred = model(x)
  print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
  print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
  
  example_batch_loss = compute_loss(y, pred)
  print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
  print("scalar_loss:      ", example_batch_loss.numpy().mean())
  
  # to get preditions from the model, we sample from output distribution. 
  # This means we are using a categorical distribution to sample over the predition.
  # this gives a predition of the next character's index at each timestep.

  sampled_indices = tf.random.categorical(pred[0], num_samples=1)
  sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
  
  print("Input: \n", str(repr("".join(io.i2c[x[0]]))))
  print()
  print("Next Char Predictions: \n", repr("".join(io.i2c[sampled_indices])))