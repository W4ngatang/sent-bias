''' Code to load InferSent '''
import os
import logging as log
import torch


INFERSENT_PATHS = {'all': 'infersent.allnli.pickle', 'snli': 'infersent.snli.pickle'}


def encode(model, sents, tokenize=True):
    ''' Use model to encode sents '''
    encs = model.encode(sents, bsize=1, tokenize=tokenize)
    sent2enc = {sent: enc for sent, enc in zip(sents, encs)}
    return sent2enc


def load_infersent(path_prefix, glove_path, train_data='all', use_cpu=False):
    ''' Load pretrained infersent model '''
    if use_cpu:
        kwargs = dict(map_location='cpu')
    else:
        kwargs = dict()
    infersent = torch.load(
        os.path.join(
            path_prefix,
            INFERSENT_PATHS[train_data]),
        **kwargs)
    if use_cpu:
        infersent.use_cuda = False
    infersent.set_glove_path(glove_path)
    log.info("Successfully loaded infersent!")
    return infersent
