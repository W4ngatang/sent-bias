# https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
# https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb

import logging
import collections
import os
import pickle

from fastai.text import get_language_model
import torch
from torch import LongTensor
from torch.autograd import Variable
from torchtext import data


PRE_LM_FILENAME = 'fwd_wt103.h5'
VOCAB_FILENAME = 'itos_wt103.pkl'
EMB_SZ = 400
N_HID = 1150
N_LAYERS = 3


def load_model(ulmfit_dir, use_cpu=False):
    pre_lm_path = os.path.join(ulmfit_dir, PRE_LM_FILENAME)
    vocab_path = os.path.join(ulmfit_dir, VOCAB_FILENAME)

    logging.info('loading vocab')
    itos = pickle.load(open(vocab_path, 'rb'))
    unk_token = itos.index('_unk_')
    pad_token = itos.index('_pad_')
    stoi = collections.defaultdict(lambda: unk_token,
                                   {v: k for k, v in enumerate(itos)})

    logging.info('loading model')
    if use_cpu:
        kwargs = dict(map_location=lambda storage, loc: storage)
    else:
        kwargs = dict()
    wgts = torch.load(pre_lm_path, **kwargs)
    model = get_language_model(len(itos), EMB_SZ, N_HID, N_LAYERS, pad_token)
    model.load_state_dict(wgts)
    model.eval()

    return (model, stoi)


def embed_sentence(model, stoi, sentence, bos=None, eos=None):
    if bos is not None:
        sentence = [bos] + sentence
    if eos is not None:
        sentence = sentence + [eos]

    t = Variable(LongTensor([[stoi[w] for w in sentence]]).view(-1, 1))

    # Reset hidden state
    model.reset()
    # Get predictions from model (outputs of LinearDecoder)
    (result, raw_outputs, outputs) = model(t)

    sentence_rep = outputs[-1].data.numpy().squeeze(axis=1)
    if bos is not None:
        sentence_rep = sentence_rep[1:]
    if eos is not None:
        sentence_rep = sentence_rep[:-1]

    return sentence_rep


def encode(sentences, ulmfit_dir, use_cpu=False):
    (model, stoi) = load_model(ulmfit_dir, use_cpu=use_cpu)
    return dict(
        (sentence, embed_sentence(model, stoi, sentence))
        for sentence in sentences
    )
