import h5py
import numpy as np

def load_elmo_hdf5(hdf5_filename):
  """
  Load a text file and corresponding elmo hdf5 file
  into a text to vector dictionary mapping
  The number of entries in the dict is the number
  of lines in the text_file.
  If a line has multiple tokens (instead of one word),
  sum into a single vector; the three layers are concatenated.
  The file was made with the --use-sentence-keys elmo flag.
  """
  text2vec = {}
  h5py_file = h5py.File(hdf5_filename, 'r')
  for key in h5py_file:
    raw = h5py_file[key] # <HDF5 dataset "zinnia": shape (3, 1, 1024), type "<f4">
    nparray = raw[()] # <class 'numpy.ndarray'>; (3, 1, 1024)
    assert len(nparray.shape) == 3
    nparray = np.mean(nparray, axis=1) # average vectors across time steps
    assert len(nparray.shape) == 2
    nparray = np.reshape(nparray, -1) # concatenate the three layers
    assert len(nparray.shape) == 1
    text2vec[key] = nparray
  return text2vec
