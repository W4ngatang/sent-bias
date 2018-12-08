''' BoW encoder '''
import io
import logging as log
import numpy as np


def get_word_dict(sentences, tokenize=True):
    ''' From sentences create vocab of words '''
    word_dict = {}
    if tokenize:
        from nltk.tokenize import word_tokenize
    else:
        word_tokenize = str.split
    tokenized_sents = [word_tokenize(s) for s in sentences]
    for sent in tokenized_sents:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    # word_dict['<s>'] = ''
    # word_dict['</s>'] = ''
    return tokenized_sents, word_dict


def get_vecs(sentences, word_vec, dim):
    ''' Create BoW representations for sentences using word_vecs '''
    bow_vec = {}
    for sent in sentences:
        key = []
        vec = np.zeros(dim)  # initialize w/ zeros

        for word in sent:
            key.append(word)
            single_wordvec = np.array(word_vec[word])
            vec += single_wordvec
        bow_vec[' '.join(key)] = vec / len(sent)

    return bow_vec


def get_glove(vocab, glove_path):
    ''' Load vectors for words in word_dict from glove_path '''
    word_vecs = {}
    dim = None
    with io.open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in vocab:
                word_vecs[word] = np.fromstring(vec, sep=' ')
                if dim is None:
                    dim = len(word_vecs[word])
                else:
                    assert len(word_vecs[word]) == dim
    log.info('Found %d/%d words with glove vectors', len(word_vecs), len(vocab))
    return word_vecs, dim


def encode(sentences, glove_path, tokenize=True):
    ''' Encode sentences into BoW representation '''
    sents, vocab = get_word_dict(sentences, tokenize=tokenize)

    word_vecs, dim = get_glove(vocab, glove_path)

    bow_word_vecs = get_vecs(sents, word_vecs, dim)

    # log.info('Vocab size : %d', len(word_vecs))
    # log.info('No of Sentences : %d', len(bow_word_vecs))
    return bow_word_vecs
