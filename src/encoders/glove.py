import numpy as np

# Glove 840B 300d at:
# /Users/rachelrudinger/data/glove/glove.840B/glove.840B.300d.txt

def load_glove_file(fname):
  """ loads a glove-styled file into a word to vector dictionary """
  word2vector = {}
  with open(fname, 'r') as fp:
    for line in fp:
      line = line.split(' ')
      word = line[0]
      vector = line[1:]
      vector = np.asarray([float(f) for f in vector])
      word2vector[word] = vector
  return word2vector
