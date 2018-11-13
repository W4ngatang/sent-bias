import ipdb
import numpy as np
import time
import io



def get_word_dict(sentences, tokenize=True):
    # create vocab of words
    word_dict = {}
    if tokenize:
        from nltk.tokenize import word_tokenize
    sentences = [s.split() if not tokenize else word_tokenize(s)
                 for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    return sentences,word_dict

def get_vecs(sentences, word_vec, glove_path):
    bow_vec={}
    for sent in sentences:
        key = ''
        vec = np.zeros(int(glove_path[-8:-5]))
        
        for word in sent:
            key = key+word+' '
            single_wordvec = np.array(word_vec[word])
            vec+=single_wordvec
        bow_vec[key[:-1]]  = vec/len(sent)
        
    return bow_vec    
    

def get_glove(word_dict, glove_path):
    
    word_vec = {}
    with io.open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.fromstring(vec, sep=' ')
    print('Found {0}(/{1}) words with glove vectors'
          .format(len(word_vec), len(word_dict)))
    return word_vec

def get_bow_vecs(sentences, glove_path, tokenize=True):
    sents, word_dict = get_word_dict(sentences, tokenize)
    word_vec = get_glove(word_dict,glove_path)
    BoW_word_vecs = get_vecs(sents,word_vec, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    print('No of Sentences : {0}'.format(len(BoW_word_vecs)))
    return BoW_word_vecs