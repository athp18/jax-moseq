import os
import joblib

from .utils import *
from .debugging import *

def save_model(model, path):
  assert os.path.exists(path)
  if path.endswith('.p') or path.endswith('.pkl'):
    joblib.dump(model, path)

def load_model(filename):
  if filename.endswith('.p') or filename.endswith('.pkl'):
    return joblib.load(filename)
