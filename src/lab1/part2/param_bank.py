#!/usr/bin/python3

default_params = {
  ### Hyperparameter setting and optimization ###
  # Optimization parameters:
  "num_training_iterations" : 2000,  # Increase this to train longer
  "batch_size" : 4,  # Experiment between 1 and 64
  "seq_length" : 100,  # Experiment between 50 and 500
  "learning_rate" : 5e-3,  # Experiment between 1e-5 and 1e-1


# Model parameters: 
  "vocab_size" : 0,
  "embedding_dim" : 256,
  "rnn_units" : 1024  # Experiment between 1 and 2048
}

mild_params = {
  "num_training_iterations" : 1000,  # Increase this to train longer
  "batch_size" : 10,  # Experiment between 1 and 64
  "seq_length" : 50,  # Experiment between 50 and 500
  "learning_rate" : 5e-3,  # Experiment between 1e-5 and 1e-1


# Model parameters: 
  "vocab_size" : 0,
  "embedding_dim" : 256,
  "rnn_units" : 256  # Experiment between 1 and 2048
}
