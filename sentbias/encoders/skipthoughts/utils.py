import _pickle as pk
import gzip
import numpy as np
import os

from sentbias.data import load_json


def preprocess_file(filepath, output_path):
    dict_of_sentences = load_json(filepath, False)
    assert len(dict_of_sentences) == 4
    tmp_output_path = output_path[:-3] + 'tmp'
    tokens = set()
    sentence_tokens = []
    MAX_SENTENCE_LENGTH = 0

    for sentences in dict_of_sentences.values():
        for j in range(len(sentences)):
            sents = sentences[j]

            sent_length = len(sents.split())
            # print(len(sents.split()))
            if sent_length > MAX_SENTENCE_LENGTH:
                MAX_SENTENCE_LENGTH = sent_length

            sent_tokens = []
            for w in sents.split():
                sent_tokens.append(w)
                tokens.add(w)
                sentence_tokens.append(sent_tokens)
    with gzip.open(tmp_output_path, 'w') as f:
        pk.dump(sentence_tokens, f)
    tokens = set()
    for sent in sentence_tokens:
        tokens.update(set(sent))
    os.remove(tmp_output_path)
    return tokens, MAX_SENTENCE_LENGTH


def read_vocab(vocab_path):
    """
    Read a vocabulary file. Returns a list of words
    """
    vocab = []
    with open(vocab_path, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))

    return vocab


def encode_sentences(sentences, word_to_idx, MAX_SENTENCE_LENGTH):
    """
    Encode tokens in sentences by vocab indices
    """
    sent2vec = {}
    encoded_sentences = []
    encoded_sentences_lengths = []
    for sent in sentences:

        i = 0
        encoder = np.zeros(MAX_SENTENCE_LENGTH).tolist()
        encoded_sentences_length = len(sent.split())
        for w in sent.split():
            encoder[i] = word_to_idx[w]
            i += 1
        encoded_sentences.append(encoder)
        encoded_sentences_lengths.append(encoded_sentences_length)
        sent2vec[sent] = encoded_sentences
    return encoded_sentences, encoded_sentences_lengths, sent2vec


def get_embedding_dictionary(sentences, sent2vec):
    ''' Use model to encode skipthougt sents '''
    assert len(sentences) == len(sent2vec)
    i = 0
    for s in sent2vec:
        sent2vec[s] = sentences[i]
        i += 1

    return sent2vec
