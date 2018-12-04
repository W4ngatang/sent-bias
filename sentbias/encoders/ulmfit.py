# Based on:
# https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
# https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb

import logging
import collections
import os
import pickle

import numpy as np
import torch
from torch import LongTensor
from torch.autograd import Variable
from torchtext import data
from fastai.text import get_language_model


PRE_LM_FILENAME = 'fwd_wt103.h5'
VOCAB_FILENAME = 'itos_wt103.pkl'
EMB_SZ = 400
N_HID = 1150
N_LAYERS = 3


def load_model(model_dir, use_cpu=False):
    pre_lm_path = os.path.join(model_dir, PRE_LM_FILENAME)
    vocab_path = os.path.join(model_dir, VOCAB_FILENAME)

    logging.info('loading vocab from {}'.format(vocab_path))
    itos = pickle.load(open(vocab_path, 'rb'))
    unk_token = itos.index('_unk_')
    pad_token = itos.index('_pad_')
    stoi = collections.defaultdict(lambda: unk_token,
                                   {v: k for k, v in enumerate(itos)})

    logging.info('loading model from {}'.format(pre_lm_path))
    if use_cpu:
        kwargs = dict(map_location=lambda storage, loc: storage)
    else:
        kwargs = dict()
    wgts = torch.load(pre_lm_path, **kwargs)
    for (old_key, new_key) in (
            ('0.encoder_with_dropout.embed.weight', '0.encoder_dp.emb.weight'),
            ('0.rnns.0.module.weight_hh_l0_raw', '0.rnns.0.weight_hh_l0_raw'),
            ('0.rnns.1.module.weight_hh_l0_raw', '0.rnns.1.weight_hh_l0_raw'),
            ('0.rnns.2.module.weight_hh_l0_raw', '0.rnns.2.weight_hh_l0_raw'),
    ):
        wgts[new_key] = wgts[old_key]
        del wgts[old_key]
    model = get_language_model(len(itos), EMB_SZ, N_HID, N_LAYERS, pad_token,
                               bias=False)
    model.load_state_dict(wgts)
    model.eval()

    return (model, stoi)


def embed_sentence(model, stoi, sentence, tokenize=False, bos=None, eos=None,
                   time_combine_method='max', layer_combine_method='add'):
    if tokenize:
        from nltk.tokenize import word_tokenize
    else:
        word_tokenize = str.split
    sentence = word_tokenize(sentence)

    if bos is not None:
        sentence = [bos] + sentence
    if eos is not None:
        sentence = sentence + [eos]

    t = Variable(LongTensor([[stoi[w] for w in sentence]]).view(-1, 1))

    # Reset hidden state
    model.reset()
    # Get predictions from model (outputs of LinearDecoder)
    # outputs is
    (result, raw_outputs, outputs) = model(t)
    print(outputs)
    raise Exception

    # TODO reverse
    if layer_combine_method == 'add':
        layer_aggregated_outputs = outputs.sum(axis=0)
    elif layer_combine_method == 'mean':
        layer_aggregated_outputs = outputs.mean(axis=0)
    elif layer_combine_method == 'concat':
        layer_aggregated_outputs = np.concatenate(outputs, axis=0)
    elif layer_combine_method == 'last':
        layer_aggregated_outputs = outputs[-1]
    else:
        raise NotImplementedError

    word_reps = layer_aggregated_outputs.data.numpy().squeeze(axis=1)
    if bos is not None:
        word_reps = word_reps[1:]
    if eos is not None:
        word_reps = word_reps[:-1]

    if time_combine_method == 'max':
        sentence_rep = word_reps.max()
    elif time_combine_method == 'mean':
        sentence_rep = word_reps.mean()
    elif time_combine_method == 'concat':
        sentence_rep = np.concatenate(word_reps, axis=0)
    elif time_combine_method == 'last':
        sentence_rep = word_reps[-1]
    else:
        raise NotImplementedError

    return sentence_rep


def encode(sentences, model_dir, use_cpu=False, tokenize=False,
           time_combine_method='max', layer_combine_method='add'):
    (model, stoi) = load_model(model_dir, use_cpu=use_cpu)
    return dict(
        (sentence, embed_sentence(model, stoi, sentence, tokenize=tokenize,
                                  time_combine_method=time_combine_method,
                                  layer_combine_method=layer_combine_method))
        for sentence in sentences
    )
