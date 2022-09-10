#!/usr/bin/python3

import tensorflow as tf 
import mitdeeplearning as mdl

songs = mdl.lab.load_training_data()
print(songs[0])