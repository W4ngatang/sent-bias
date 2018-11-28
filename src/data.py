''' Helper functions related to loading and saving data '''
import os
from collections import defaultdict
import numpy as np
import h5py

WEAT_SETS = ["targ1", "targ2", "attr1", "attr2"]

def load_sents(sent_file, split_sentence_into_list=True):
    ''' Load sentences from sent_file.  Exact format will change a lot.

    args:
        - sent_file
        - split_sentence_into_list (Bool): if True, split sentence into List(str).
            Otherwise, leave a sentence as one long string, as
            some models internally do their own tokenization

    returns:
        - data (Dict[List[str]]): dictionary containing four word sets'''

    data = defaultdict(list)
    with open(sent_file, 'r') as sent_fh:
        for row in sent_fh:
            if row[0] == "#":
                continue
            role, _ , examples = row.strip().split(':')
            assert role in WEAT_SETS
            if split_sentence_into_list:
                data[role] += [e.split() for e in examples.split(',')]
            else:
                data[role] += examples.split(',')

    for weat_set in WEAT_SETS: # check that all the needed word sets are there
        assert weat_set in data
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
                split[ex] = enc
    return


def load_jiant_encodings(enc_file, n_header=1, is_openai=False):
    ''' Load a dumb tsv format of jiant encodings.
    This is really brittle and makes a lot of assumptions about ordering. '''
    encs = []
    last_cat = None
    with open(enc_file, 'r') as enc_fh:
        for _ in range(n_header):
            enc_fh.readline() # header
        for row in enc_fh:
            idx, category, string, enc = row.strip().split('\t')
            if is_openai:
                string = " ".join([w.rstrip("</w>") for w in string])
            enc = [float(n) for n in enc[1:-1].split(',')]
            #encs[category][string] = np.array(enc)
            if last_cat is None or last_cat != category:
                #encs.append([np.array(enc)])
                encs.append({string: np.array(enc)})
            else:
                #encs[-1].append(np.array(enc))
                encs[-1][string] = np.array(enc)
            last_cat = category
            #encs[category].append(np.array(enc))

    return encs
