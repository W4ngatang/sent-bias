import sys
import numpy as np

# Glove 840B 300d at:
# /Users/rachelrudinger/data/glove/glove.840B/glove.840B.300d.txt

def load_word_set(fname):
  word_set = set({})
  with open(fname, 'r') as fp:
    for line in fp:
      word_set.add(line.strip())
  return word_set

def load_weat_test_words(fname):
  concept_wordset_list = []
  with open(fname, 'r') as fp:
    for line in fp:
      line = line.strip().split(':')
      assert len(line) == 2
      concept, words = line[0], line[1]
      word_set = set(words.split(','))
      concept_wordset_list.append((concept, word_set))
  assert len(concept_wordset_list) == 4
  WEAT = {}
  WEAT["X"] = concept_wordset_list[0]
  WEAT["Y"] = concept_wordset_list[1]
  WEAT["A"] = concept_wordset_list[2]
  WEAT["B"] = concept_wordset_list[3]
  return WEAT

def create_vec_set(word_set, glove_path=None):
  """
  Takes a set of words, finds the corresponding glove vectors
  returns a list of strings, to print.
  """
  if glove_path is None:
    glove_path = "/Users/rachelrudinger/data/glove/glove.840B/glove.840B.300d.txt"
  assert len(word_set) > 0
  out_lines = []
  with open(glove_path, 'r') as fp:
    for line in fp:
      word = line[:line.find(' ')]
      if word in word_set:
        out_lines.append(line)
        #sys.stdout.write(line)
        word_set.remove(word)
        if len(word_set) == 0:
          break
  return out_lines

def create_weat_vec_files(weatfile, outpath=None, glove_path=None):
  weatname = weatfile.split('/')[-1][:-4] # remove .txt
  if outpath is None:
    outpath = "../tests/"
  WEAT = load_weat_test_words(weatfile)
  for s in ["A","B","X","Y"]:
    concept, word_set = WEAT[s]
    out_lines = create_vec_set(word_set, glove_path=glove_path)
    with open(outpath+weatname+"."+s+".vec", 'w') as fp:
      for o in out_lines:
        fp.write(o)

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
