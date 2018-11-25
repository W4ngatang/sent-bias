''' Code to load InferSent '''
import os
import logging as log
import torch

INFERSENT_PATHS = {'all':'infersent.allnli.pickle', 'snli':'infersent.snli.pickle'}

def encode(model, sents):
    ''' Use model to encode sents '''
    encs = model.encode(sents, bsize=1, tokenize=False)
    sent2enc = {sent: enc for sent, enc in zip(sents, encs)}
    return sent2enc

def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings

def load_infersent(path_prefix, glove_path, train_data='all'):
    ''' Load pretrained infersent model '''
    infersent = torch.load(os.path.join(path_prefix, INFERSENT_PATHS[train_data]))#, map_location='cpu') # TODO(Alex): make this an option
    infersent.set_glove_path(glove_path)
    log.info("Successfully loaded infersent!")
    return infersent
