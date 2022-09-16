#!/usr/bin/python3
import tensorflow as tf
import mitdeeplearning as mdl

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from p1_1_mnist import data_loader

def build_cnn_model():
  model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=24, kernel_size=(3,3), activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            tf.keras.layers.Conv2D(filters=36, kernel_size=(3,3),activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
          ])
  return model

def model_example(train_images):
  cnn_model = build_cnn_model()
  cnn_model.predict(train_images[[0]])
  print(cnn_model.summary())


# epoch is an iteration over the entire x and y data provided
EPOCHS = 5

# number of samples per gradient update
BATCH_SIZE = 64
def cnn_builder():
  cnn_model = build_cnn_model()
  cnn_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return cnn_model

def cnn_trainer(cnn_model, train_images, train_labels):
  cnn_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

def cnn_evaluator(cnn_model, test_images, test_labels):
  test_loss, test_acc = cnn_model.evaluate(test_images, test_labels, batch_size=BATCH_SIZE)
  print("Test accuracy: ", test_acc)
  print(cnn_model.summary())

def main():
  
  train_images, train_labels, test_images, test_labels = data_loader()
  cnn_model = cnn_builder()
  cnn_trainer(cnn_model, train_images, train_labels)
  cnn_evaluator(cnn_model, test_images, test_labels)

  # model_example(train_images)

main()

