#!/usr/bin/python3

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

def main():
  download_mnist()

main()
