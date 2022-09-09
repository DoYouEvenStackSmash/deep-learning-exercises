#!/usr/bin/python3

import numpy as np

def text_vectorize_prep(vocab):
  ### Define numerical representation of text ###

  # Create a mapping from character to unique index.
  # For example, to get the index of the character "d", 
  #   we can evaluate `char2idx["d"]`.  
  char2idx = {u:i for i, u in enumerate(vocab)}

  # Create a mapping from indices to characters. This is
  #   the inverse of char2idx and allows us to convert back
  #   from unique index to the character in our vocabulary.
  idx2char = np.array(vocab)
  ''' now we have our pair mapping
  char2idx: char -> index in idx2char
  idx2char: index -> char at idx2char index
  '''
  return char2idx, idx2char

  # print('{')
  # for char,_ in zip(char2idx, range(20)):
  #   print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
  # print('  ...\n}')

'''
  Map characters to integer alphabet

  Return numpy array of integers
'''  
def vectorize_string(all_songs_string, char2idx_dict, idx2char_arr = None):
  np_str_arr = []
  for i in range(len(all_songs_string)):
    np_str_arr.append(char2idx_dict[all_songs_string[i]])
  return np.array(np_str_arr)

'''
  Input a very long string of all songs, concatenated.
  
  Return the integer analogue of the very long string, and the 
  supporting data for mapping between int and char
'''
def vectorize_songs(concat_songs):
  unique_notes = sorted(set(concat_songs))
  c2i_dict,i2c_arr = text_vectorize_prep(unique_notes)
  vectorized_songs = vectorize_string(concat_songs, c2i_dict)
  assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"
  return vectorized_songs, c2i_dict, i2c_arr