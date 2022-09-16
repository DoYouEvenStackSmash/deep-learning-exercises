#!/usr/bin/python3

from calendar import EPOCH
import tensorflow as tf
import mitdeeplearning as mdl

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

def download_mnist():
  mnist = tf.keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  train_images = (np.expand_dims(train_images, axis = -1)/255.).astype(np.float32)
  train_labels = (train_labels).astype(np.int64)
  test_images = (np.expand_dims(test_images, axis = -1)/255.).astype(np.float32)
  test_labels = (test_labels).astype(np.int64)
  plt.figure(figsize=(10,10))
  random_inds = np.random.choice(60000,36)
  for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])
  plt.show()

'''
  On softmax:
  Softmax converts a vector of values to a probability distribution.

  The elements of the output vector are in range (0, 1) and sum to 1.

  Each vector is handled independently. The axis argument sets which axis of the input the function is applied along.

  Softmax is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.
'''

def build_fc_model():
  fc_model = tf.keras.Sequential([
    # flatten the 28x28 image into a single 784 element vector
    tf.keras.layers.Flatten(),

    # define the activation function for the first fully connected dense layer (best as i can tell, we want ReLu?)
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dense layer to output probabilities for 0-9. The array should sum to 1
    # see note on softmax above
    tf.keras.layers.Dense(10, activation='softmax') 
  ])
  return fc_model

# cross entropy loss: for models(like ours) that output probability between 0 and 1
# note, to update the model later, we'll need to re run the above cell
def model_builder():
  model = build_fc_model()
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model


  

  

def data_loader():
  mnist = tf.keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  train_images = (np.expand_dims(train_images, axis = -1)/255.).astype(np.float32)
  train_labels = (train_labels).astype(np.int64)
  test_images = (np.expand_dims(test_images, axis = -1)/255.).astype(np.float32)
  test_labels = (test_labels).astype(np.int64)
  return train_images, train_labels, test_images, test_labels


# epoch is an iteration over the entire x and y data provided
EPOCHS = 5

# number of samples per gradient update
BATCH_SIZE = 64

# TRAIN THE COMPILED MODEL
def model_trainer(model, train_images, train_labels):
  model.fit(train_images,train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
  # return model

#evaluate model accuracy
def model_evaluator(model, test_images, test_labels):
  test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
  print('Test Accuracy:',test_acc)


def main():
  tr_im, tr_la, te_im, te_la = data_loader()
  model = model_builder()
  model_trainer(model, tr_im, tr_la)
  model_evaluator(model, te_im, te_la)
  
  # download_mnist()

main()
