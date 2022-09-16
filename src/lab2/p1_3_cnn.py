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


def get_confidence(predictions,index=0):
  print("first image prediction: {}".format(predictions[index]))
  max_confidence = -1
  digit = -1
  for i in range(len(predictions[0])):
    if predictions[0][i] > max_confidence:
      digit = i
      max_confidence = predictions[0][i]
  print("highest confidence digit {}:\t{}".format(digit, max_confidence))

def cnn_predictions(cnn_model, test_images):
  predictions = cnn_model.predict(test_images)
  return predictions
  
def visualize_classification_results(predictions, test_images, test_labels):
  #@title Change the slider to look at the model's predictions! { run: "auto" }
  image_index = 79 #@param {type:"slider", min:0, max:100, step:1}
  plt.subplot(1,2,1)
  mdl.lab2.plot_image_prediction(image_index, predictions, test_labels, test_images)
  plt.subplot(1,2,2)
  mdl.lab2.plot_value_prediction(image_index, predictions,  test_labels)
  plt.show()

def show_many_images(predictions, test_images, test_labels, num_images):
  num_rows = 5
  num_cols = 4
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    mdl.lab2.plot_image_prediction(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    mdl.lab2.plot_value_prediction(i, predictions, test_labels)
  plt.show()

def data_viewer(test_images, test_labels, index=0):
  print("label of digit: {}".format(test_labels[index]))
  plt.imshow(test_images[0,:,:,0],cmap=plt.cm.binary)
  plt.show()

def visualization_driver():
  train_images, train_labels, test_images, test_labels = data_loader()
  cnn_model = cnn_builder()
  cnn_trainer(cnn_model, train_images, train_labels)
  p = cnn_predictions(cnn_model, test_images)
  # visualize_classification_results(p, test_images, test_labels)
  show_many_images(p, test_images, test_labels, 40)


def main():
  visualization_driver()
  return
  train_images, train_labels, test_images, test_labels = data_loader()
  # print(len(test_images))
  # data_viewer(test_images, test_labels)
  
  cnn_model = cnn_builder()
  cnn_trainer(cnn_model, train_images, train_labels)
  cnn_predictions(cnn_model, test_images)
  # cnn_evaluator(cnn_model, test_images, test_labels)

  # model_example(train_images)

main()

