#!/usr/bin/python3
import numpy as np
import json
class InputObj:
  def __init__(self, S = None):
    self.S = S
    self.vocab = None
    self.c2i = None
    self.i2c = None
    self.V = None
  
  def process_strings(self):
    self.S = "\n\n".join(self.S)
    self.vocab = sorted(set(self.S))
    self.c2i = {u:i for i, u in enumerate(self.vocab)}
    self.i2c = np.array(self.vocab)
    self.V = np.array([self.c2i[c] for c in self.S])


