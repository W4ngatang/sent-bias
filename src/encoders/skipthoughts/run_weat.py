''' Main script for loading models and running WEAT tests '''
import os
import sys
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
import h5py # pylint: disable=import-error
import nltk
import ipdb
import numpy as np
import skipthoughts
from utils import *
import weat


TESTS = ["weat1", "weat2", "weat3", "weat4"]
MODELS = ["skipthoughts"]
def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--log_file', '-l', type=str, help="File to log to")
    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test")
    parser.add_argument('--exp_dir', '-sd', type=str,
                        help="Directory from which to load and save vectors. " +
                        "Files should be stored as h5py files.")
    parser.add_argument('--dir_st', '-g', type=str, help="Skipthought Model path")
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
 
    
 
def main(arguments):
    ''' Main logic: parse args for tests to run and which models to evaluate '''
    args = handle_arguments(arguments)
    maybe_make_dir(args.exp_dir)
    log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    tests = split_comma_and_check(args.tests, TESTS, "test")
    models = split_comma_and_check(args.models, MODELS, "model")
    model_name = "skipthoughts"
    encsA, encsB, encsX, encsY ={},{},{},{}
    for test in tests:
        enc_file = os.path.join(args.exp_dir, "%s.%s.h5" % (model_name, test))
        encs = load_encodings(enc_file)

        if encs is None:
            
            log.info("Unable to find saved encodings for model %s for test %s. " +
                         "Generating new encodings.", model_name, test)
            
            filepath = os.path.join(args.data_dir, "%s.txt" % test)
            vocab_path='/home/sb6416/sentbias/data/Vocab.txt'
            output_path = '/home/sb6416/data'
            vocab=set()
            tokens,MAX_SENTENCE_LENGTH = preprocess_file(filepath, output_path)
            vocab.update(tokens)
            vocab =list(vocab)

            with open(vocab_path, 'w') as f:
                f.write('\n'.join(vocab))
            word_to_idx = {w: idx for (idx, w) in enumerate(read_vocab(vocab_path))}     

            array_of_sentences = load_single_word_sents(filepath)
            encoded_sentences=[]
            log.info("Encoding sentences for test %s with model %s...", test, model_name)
                   
            for i in range(0,len(array_of_sentences)):
                sentences = array_of_sentences[i]
                sentences = encode_sentences(sentences, word_to_idx,MAX_SENTENCE_LENGTH)
                encoded_sentences.append(sentences)
        
            input = Variable(torch.tensor(encoded_sentences)).type(torch.LongTensor) 
            model = skipthoughts.BiSkip(args.dir_st, vocab)
        
            model.eval()
            #print(input[0])
            #wordvecs = model(input[0])
        
            encsA = model(input[0])
            encsB = model(input[1])
            encsX = model(input[2])
            encsY = model(input[3])
            
            
            all_encs = [encsA, encsB, encsX, encsY]
            log.info("\tDone!")

                    # Save everything
            save_encodings(all_encs, enc_file)
            log.info("Saved encodings for model %s to %s", model_name, enc_file)   
        
        else:
            encsA, encsB, encsX, encsY = encs
        
        log.info("Running test %s on %s", test, model_name)
            
        weat.run_test(encsA, encsB, encsX, encsY)
            
       
if __name__ == "__main__":
    main(sys.argv[1:])
    