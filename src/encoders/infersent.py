''' Code to load InferSent '''
import os
import logging as log
import torch

PATH_PREFIX = '/scratch/sb6416/senteval/infersent/encoder'
INFERSENT_PATHS = {'all':'infersent.allnli.pickle', 'snli':'infersent.snli.pickle'}
for k, v in INFERSENT_PATHS.items():
    INFERSENT_PATHS[k] = "%s/%s" % (PATH_PREFIX, v)

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

def load_infersent(glove_path, train_data='all'):
    ''' Load pretrained infersent model '''
    infersent = torch.load(INFERSENT_PATHS[train_data])#, map_location='cpu') # TODO(Alex): make this an option
    infersent.set_glove_path(glove_path)
    log.info("Successfully loaded infersent!")
    return infersent
