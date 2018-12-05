# Based on:
# https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
# https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb

import logging as log
import collections
import os
import pickle

import numpy as np
import torch
from torch import LongTensor
from torch.autograd import Variable
import spacy

from .text import get_language_model


PRE_LM_FILENAME = 'fwd_wt103.h5'
VOCAB_FILENAME = 'itos_wt103.pkl'
EMB_SZ = 400
N_HID = 1150
N_LAYERS = 3

SPACY_EN = spacy.load('en')


def spacy_tokenize(s):
    return [token.text for token in SPACY_EN(s)]


class ULMFiT(object):
    def __init__(self, model_dir, use_cpu=False):
        pre_lm_path = os.path.join(model_dir, PRE_LM_FILENAME)
        vocab_path = os.path.join(model_dir, VOCAB_FILENAME)

        log.info('loading vocab from {}'.format(vocab_path))
        itos = pickle.load(open(vocab_path, 'rb'))
        unk_token = itos.index('_unk_')
        pad_token = itos.index('_pad_')
        self.stoi = collections.defaultdict(lambda: unk_token,
                                            {v: k for k, v in enumerate(itos)})

        log.info('loading model from {}'.format(pre_lm_path))
        if use_cpu:
            kwargs = dict(map_location=lambda storage, loc: storage)
        else:
            kwargs = dict()
        wgts = torch.load(pre_lm_path, **kwargs)
        self.model = get_language_model(len(itos), EMB_SZ, N_HID, N_LAYERS, pad_token)
        self.model.load_state_dict(wgts)
        self.model.eval()

    def embed_sentence(self, sentence, tokenize=False, bos=None, eos=None,
                       time_combine_method='max', layer_combine_method='add'):
        if tokenize:
            word_tokenize = spacy_tokenize
        else:
            word_tokenize = str.split
        sentence = word_tokenize(sentence)

        if bos is not None:
            sentence = [bos] + sentence
        if eos is not None:
            sentence = sentence + [eos]

        t = Variable(LongTensor([[self.stoi[w] for w in sentence]]).view(-1, 1))

        # Reset hidden state
        self.model.reset()
        # Get predictions from model (outputs of LinearDecoder).
        # `outputs` is a list of tensors of sizes
        # num_words x batch_size x num_hidden, corresponding to the
        # layers of the underlying RNN.  Note that the layers have
        # different sizes, namely 1150, 1150, 400
        (result, raw_outputs, outputs) = self.model(t)

        np_outputs = [o.data.numpy().squeeze(axis=1) for o in outputs]

        if bos is not None:
            np_outputs = [o[1:] for o in np_outputs]
        if eos is not None:
            np_outputs = [o[:-1] for o in np_outputs]

        if time_combine_method == 'max':
            layer_reps = [o.max(axis=0) for o in np_outputs]
        elif time_combine_method == 'mean':
            layer_reps = [o.mean(axis=0) for o in np_outputs]
        elif time_combine_method == 'concat':
            layer_reps = [np.reshape(o, -1) for o in np_outputs]
        elif time_combine_method == 'last':
            layer_reps = [o[-1] for o in np_outputs]
        else:
            raise NotImplementedError

        if layer_combine_method == 'last':
            sentence_rep = layer_reps[-1]
        elif layer_combine_method == 'concat':
            sentence_rep = np.concatenate(layer_reps)
        else:
            raise NotImplementedError

        return sentence_rep

    def encode(self, sentences, tokenize=False,
               time_combine_method='max', layer_combine_method='add'):
        return dict(
            (sentence, self.embed_sentence(sentence, tokenize=tokenize,
                                           time_combine_method=time_combine_method,
                                           layer_combine_method=layer_combine_method))
            for sentence in sentences
        )
