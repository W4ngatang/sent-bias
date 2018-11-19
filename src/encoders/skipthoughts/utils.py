from nltk.tokenize import word_tokenize
import numpy as np
import torch
from torch.autograd import Variable
import os
import sys
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
import h5py # pylint: disable=import-error
import nltk
import ipdb
import numpy as np
import gzip
import _pickle as pk



def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--log_file', '-l', type=str, help="File to log to")
    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test")
    parser.add_argument('--exp_dir', '-sd', type=str,
                        help="Directory from which to load and save vectors. " +
                        "Files should be stored as h5py files.")
    parser.add_argument('--glove_path', '-g', type=str, help="File to GloVe vectors")
    parser.add_argument('--ignore_cached_encs', '-i', type=str,
                        help="1 if ignore existing encodings and encode from scratch")

    parser.add_argument('--tests', '-t', type=str, help="WEAT tests to run")
    parser.add_argument('--models', '-m', type=str, help="Model to evaluate")
    return parser.parse_args(arguments)


def split_comma_and_check(arg_str, allowed_set, item_type):
    ''' Given a comma-separated string of items,
    split on commas and check if all items are in allowed_set.
    item_type is just for the assert message. '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError("Unknown %s: %s!" % (item_type, item))
    return items


def maybe_make_dir(dirname):
    ''' Maybe make directory '''
    os.makedirs(dirname, exist_ok=True)



    
def load_single_word_sents(sent_file):
    ''' Load sentences from sent_file.
    Exact format will change a lot. '''
    data = []
    with open(sent_file, 'r') as sent_fh:
        for row in sent_fh:
            _, examples = row.strip().split(':')
            
            data.append(examples.split(','))
            
    return data


def load_encodings(enc_file):
    ''' Search to see if we already dumped the vectors for a model somewhere
    and return it, else return None. '''
    if not os.path.exists(enc_file):
        return None
    encs = []
    with h5py.File(enc_file, 'r') as enc_fh:
        for split_name, split in enc_fh.items():
            split_d = {}
            for ex, enc in split.items():
                split_d[ex] = enc[:]
            encs.append(split_d)
    return encs


def save_encodings(encodings, enc_file):
    ''' Save encodings to file '''
    with h5py.File(enc_file, 'w') as enc_fh:
        for split_name, split_encodings in zip(['A', 'B', 'X', 'Y'], encodings):
            split = enc_fh.create_group(split_name)
            for ex, enc in split_encodings.items():
                split['%s' % ex] = enc
    return

def preprocess_file(filepath, output_path):
    with open(filepath, 'r') as f:
        text = f.read()
    array_of_sentences = load_single_word_sents(filepath)
    assert len(array_of_sentences) == 4 
    tmp_output_path = output_path[:-3] + 'tmp'
    tokens = set()
    sentence_tokens = []
    MAX_SENTENCE_LENGTH = 0
    
    for i in range(0,len(array_of_sentences)):
        sentences = array_of_sentences[i]
        for j in range(len(sentences)):
            sents = sentences[j]
            
            sent_length =len(sents.split())
            #print(len(sents.split()))
            if sent_length > MAX_SENTENCE_LENGTH:
                MAX_SENTENCE_LENGTH= sent_length
                
            sent_tokens = []
            for w in sents.split():
                sent_tokens.append(w)
                tokens.add(w)
                sentence_tokens.append(sent_tokens)
    with gzip.open(tmp_output_path, 'w') as f:
        pk.dump(sentence_tokens,f)
    tokens = set()
    for sent in sentence_tokens:
        tokens.update(set(sent))
    os.remove(tmp_output_path)    
    return tokens,MAX_SENTENCE_LENGTH
        
def load_single_word_sents(sent_file):
    ''' Load sentences from sent_file.
    Exact format will change a lot. '''
    data = []
    with open(sent_file, 'r') as sent_fh:
        for row in sent_fh:
            _, examples = row.strip().split(':')
            data.append(examples.split(','))
    return data 


def read_vocab(vocab_path):
    """
    Read a vocabulary file. Returns a list of words
    """
    vocab = []
    with open(vocab_path, 'r') as f:
        for line in f:
            vocab.append(line.strip('\n'))

    return vocab

def encode_sentences(sentences, word_to_idx,MAX_SENTENCE_LENGTH):
    """
    Encode tokens in sentences by vocab indices
    """
    sent2vec={}
    encoded_sentences =[]
    encoded_sentences_lengths=[]
    for sent in sentences:
        
        i =0
        encoder = np.zeros(MAX_SENTENCE_LENGTH).tolist()
        encoded_sentences_length=len(sent.split())
        for w in sent.split():
            encoder[i]=word_to_idx[w]
            i+=1
        encoded_sentences.append(encoder)
        encoded_sentences_lengths.append(encoded_sentences_length)
        sent2vec[sent]=encoded_sentences
    return encoded_sentences,encoded_sentences_lengths,sent2vec


def get_embedding_dictionary(sentences, sent2vec):
    ''' Use model to encode skipthougt sents '''
    assert len(sentences) == len(sent2vec)
    i=0
    for s in sent2vec:
        sent2vec[s]= sentences[i]
        i+=1
        
    return sent2vec