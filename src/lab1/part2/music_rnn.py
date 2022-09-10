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
from vectorize_functions import *
from model_predict_examples import *

# Check that we are using a GPU, if not switch runtimes
#   using Runtime > Change Runtime Type > GPU

# assert len(tf.config.list_physical_devices('GPU')) > 0

class M:
  def __init__(self):

    ### Hyperparameter setting and optimization ###

    # Optimization parameters:
    self.num_training_iterations = 100  # Increase this to train longer
    self.batch_size = 32  # Experiment between 1 and 64
    self.seq_length = 100  # Experiment between 50 and 500
    self.learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

    # Model parameters: 
    self.embedding_dim = 256 
    self.rnn_units = 512  # Experiment between 1 and 2048
    self.vectorized_songs = None
    # Checkpoint location: 
    self.checkpoint_dir = './training_checkpoints'
    self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "my_ckpt")
    self.optimizer = None
    self.model = None
    self.vocab = None
    self.i2c_a = None
    self.c2i_d = None

m = M()


def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

# generic model constructor
'''TODO: Add LSTM and Dense layers to define the RNN model using the Sequential API.'''
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #   of a fixed embedding size
    
    #  Input layer, trainable lookup table that maps the numbers of each character to a 
    #  vector with embedding_dim dimensions. cat_one representation
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #   into the vocabulary size. 
    # this transforms some input Tensor into a 1-D tensor
    Dense(vocab_size)
  ])

  return model

### Define optimizer and training operation ###

@tf.function
def train_step(x, y): 
  # Use tf.GradientTape()
  with tf.GradientTape() as tape:
  
    '''TODO: feed the current input into the model and generate predictions'''
    # y_hat = predition
    y_hat = m.model(x)
  
    '''TODO: compute the loss!'''
    # loss is truth y against predition y_hat
    loss = compute_loss(y, y_hat)

  # Now, compute the gradients 
  '''TODO: complete the function call for gradient computation. 
      Remember that we want the gradient of the loss with respect all 
      of the model parameters. 
      HINT: use `model.trainable_variables` to get a list of all model
      parameters.'''
  grads = tape.gradient(loss, m.model.trainable_variables)
  
  # Apply the gradients to the optimizer so it can update the model accordingly
  m.optimizer.apply_gradients(zip(grads, m.model.trainable_variables))
  return loss


def training():
  '''TODO: instantiate a new model for training using the `build_model`
    function and the hyperparameters created above.'''
  m.model = build_model(len(m.vocab), m.embedding_dim, m.rnn_units, m.batch_size)

  '''TODO: instantiate an optimizer with its learning rate.
    Checkout the tensorflow website for a list of supported optimizers.
    https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/
    Try using the Adam optimizer to start.'''
  m.optimizer = tf.keras.optimizers.Adam(m.learning_rate)

  
  ##################
  # Begin training!#
  ##################

  history = []
  plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
  if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

  for iter in tqdm(range(m.num_training_iterations)):

    # Grab a batch and propagate it through the network
    x_batch, y_batch = get_batch(m.vectorized_songs, m.seq_length, m.batch_size)
    # y_batch = truth
    loss = train_step(x_batch, y_batch)

    # Update the progress bar
    history.append(loss.numpy().mean())
    plotter.plot(history)

    # Update the model with the changed weights!
    if iter % 100 == 0:     
      m.model.save_weights(m.checkpoint_prefix)
      
  # Save the trained model and the weights
  m.model.save_weights(m.checkpoint_prefix)

# build initial model from global M m
def build_inference_model():
  m.model = build_model(len(m.vocab), m.embedding_dim, m.rnn_units, batch_size=1)
  m.model.load_weights(tf.train.latest_checkpoint(m.checkpoint_dir))
  m.model.build(tf.TensorShape([1, None]))
  m.model.summary()


'''
# Build a simple model with default hyperparameters. You will get the 
#   chance to change these later.
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
'''
# contrived model construction
def sample_model_build(vocab):
  model = build_model(len(vocab), embedding_dim=256, rnn_units=512, batch_size=32)
  model.summary()
  return model


### Prediction of a generated song ###

def generate_text(model, start_string, generation_length=1000):
  # Evaluation step (generating ABC text using the learned RNN model)

  '''TODO: convert the start string to numbers (vectorize)'''
  input_eval = ['''TODO''']
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      '''TODO: evaluate the inputs and generate the next character predictions'''
      predictions = model('''TODO''')
      
      # Remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      
      '''TODO: use a multinomial distribution to sample'''
      predicted_id = tf.random.categorical('''TODO''', num_samples=1)[-1,0].numpy()
      
      # Pass the prediction along with the previous hidden state
      #   as the next inputs to the model
      input_eval = tf.expand_dims([predicted_id], 0)
      
      '''TODO: add the predicted character to the generated text!'''
      # Hint: consider what format the prediction is in vs. the output
      text_generated.append('''TODO''')
    
  return (start_string + ''.join(text_generated))


### Defining the loss function ### 2.5

'''TODO: define the loss function to compute and return the loss between
    the true labels and predictions (logits). Set the argument from_logits=True.'''
def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) # TODO
  return loss

'''TODO: compute the loss using the true next characters from the example batch 
    and the predictions from the untrained model several cells above'''

def loss_prediction(model, vectorized_songs, c2i_d = None, i2c_a = None):
  x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
  pred = model(x)
  example_batch_loss = compute_loss(y, pred)
  print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
  print("scalar_loss:      ", example_batch_loss.numpy().mean())


def model_training():
  songs = mdl.lab1.load_training_data()
  # Join our list of song strings into a single string containing all songs
  songs_joined = "\n\n".join(songs) 
  # m = M()
  vs, c2i_d, i2c_a = vectorize_songs(songs_joined)
  m.vocab = i2c_a
  m.vectorized_songs = vs
  m.c2i_d = c2i_d
  m.i2c_a = i2c_a
  training()

def inference_model():
  songs = mdl.lab1.load_training_data()
  # Join our list of song strings into a single string containing all songs
  songs_joined = "\n\n".join(songs) 
  # m = M()
  vs, c2i_d, i2c_a = vectorize_songs(songs_joined)
  m.vocab = i2c_a
  m.vectorized_songs = vs
  m.c2i_d = c2i_d
  m.i2c_a = i2c_a
  build_inference_model()

def main():
  # model_training()
  # return
  songs = mdl.lab1.load_training_data()
  # Join our list of song strings into a single string containing all songs
  songs_joined = "\n\n".join(songs) 
  vs, c2i_d, i2c_a = vectorize_songs(songs_joined)
  m = sample_model_build(i2c_a)
  # test_prediction(m, vs, c2i_d,i2c_a)
  untrained_prediction(m, vs, c2i_d,i2c_a)
  # loss_prediction(m, vs, c2i_d,i2c_a)
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