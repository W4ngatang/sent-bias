"""Convert raw GloVe word vector text file to h5."""
import h5py
import numpy as np

embeddings_index ={}

f = open('/scratch/sb6416/senteval/gensen/data/embedding/glove.840B.300d.txt')#, encoding='utf8')
for line in f:
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.asarray(values[-300:], dtype='float32')
    embeddings_index[word] = coefs#.decode()
f.close()
vocab, vector= [],[]
for emb in embeddings_index:
    #print(emb)
    vocab.append(emb)
    vector.append(embeddings_index[emb])
vector=np.asarray(vector) 

f = h5py.File('/scratch/sb6416/senteval/gensen/data/embedding/glove.840B.300d.h5', 'w')
dt = h5py.special_dtype(vlen=str)     # PY3
f.create_dataset(data=vector, name='embedding')

voc = [v.encode('utf-8') for v in vocab]
f.create_dataset(data=voc, name='words_flatten',dtype=dt)
f.close()
