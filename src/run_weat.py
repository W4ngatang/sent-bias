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
import glove
import weat
import encoders.infersent as infersent
import encoders.gensen as gensen
import encoders.bow as bow

TESTS = ["weat1", "weat2", "weat3", "weat4"]
MODELS = ["glove", "infersent", "elmo","gensen","bow"]

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

def encode_sentences(model, sents):
    ''' Use model to encode sents '''
    encs = model.encode(sents, bsize=1, tokenize=False)
    sent2enc = {sent: enc for sent, enc in zip(sents, encs)}
    return sent2enc

def encode_sentences_gensen(model, sents):
    ''' Use model to encode gensen sents '''
    sent2vec={}
    reps_h, reps_h_t = model.get_representation(sents, pool='last', return_numpy=True,tokenize =True)
    for j in range(0,len(sents)):
        sent2vec[sents[j]]=reps_h_t[j]
    return sent2vec

def main(arguments):
    ''' Main logic: parse args for tests to run and which models to evaluate '''
    args = handle_arguments(arguments)
    maybe_make_dir(args.exp_dir)
    log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    #for wt in weat_tests:
    #  glove.create_weat_vec_files("../tests/"+wt+".txt")
    tests = split_comma_and_check(args.tests, TESTS, "test")
    models = split_comma_and_check(args.models, MODELS, "model")
    encsA, encsB, encsX, encsY ={},{},{},{}
    for model_name in models:
        ''' Different models have different interfaces for things, but generally want to:
         - if saved vectors aren't there:
            - load the model
            - load the test data
            - encode the vectors
            - dump the files into some storage
        - else load the saved vectors '''

        model = None

        for test in tests:
            enc_file = os.path.join(args.exp_dir, "%s.%s.h5" % (model_name, test))
            encs = load_encodings(enc_file)

            if encs is None:
                log.info("Unable to find saved encodings for model %s for test %s. " +
                         "Generating new encodings.", model_name, test)

                # load the test data
                sents = load_single_word_sents(os.path.join(args.data_dir, "%s.txt" % test))
                
                assert len(sents) == 4

                # load the model and do model-specific encoding procedure
                # TODO(Alex): might want to build models once (knowing what tasks to run) to avoid
                #             costly model building
                if model_name == 'elmo': # TODO(Alex): move this
                    encsA, encsB, encsX, encsY = weat.load_elmo_weat_test(test, path='elmo/')
                elif model_name == 'glove': # TODO(Alex): move this
                    encsA, encsB, encsX, encsY = weat.load_weat_test(test, path=args.data_dir)
                elif model_name == 'infersent':
                    model = infersent.load_infersent(args.glove_path, train_data='all') if model \
                            is None else model
                    model.build_vocab([s for s in sents[0] + sents[1] + sents[2] + sents[3]], tokenize=False)
                    log.info("Encoding sentences for test %s with model %s...", test, model_name)
                    encsA = encode_sentences(model, sents[0])
                   
                   
                    encsB = encode_sentences(model, sents[1])
                    encsX = encode_sentences(model, sents[2])
                    encsY = encode_sentences(model, sents[3])
                    all_encs = [encsA, encsB, encsX, encsY]
                    log.info("\tDone!")

                    # Save everything
                    save_encodings(all_encs, enc_file)
                    log.info("Saved encodings for model %s to %s", model_name, enc_file)
                    
                elif model_name =='gensen':
                    sent={}
                    model = gensen.GenSenSingle(model_folder='/scratch/sb6416/senteval/gensen/data/models',
                                         filename_prefix='nli_large_bothskip',                                                  pretrained_emb='/scratch/sb6416/senteval/gensen/data/embedding/glove.840B.300d.h5')
                    
                    encsA = encode_sentences_gensen(model, sents[0])
                    encsB = encode_sentences_gensen(model, sents[1])
                    encsX = encode_sentences_gensen(model, sents[2])
                    encsY = encode_sentences_gensen(model, sents[3])
                    
                    
                    all_encs = [encsA, encsB, encsX, encsY]
                    log.info("\tDone!")

                    # Save everything
                    save_encodings(all_encs, enc_file)
                    log.info("Saved encodings for model %s to %s", model_name, enc_file) 
                    
                elif model_name =='bow':  
                    
                    encsA = bow.get_bow_vecs(sents[0],args.glove_path)
                    encsB = bow.get_bow_vecs(sents[1],args.glove_path)
                    encsX = bow.get_bow_vecs(sents[2],args.glove_path)
                    encsY = bow.get_bow_vecs(sents[3],args.glove_path)
                                                                                  
                    all_encs = [encsA, encsB, encsX, encsY]
                    log.info("\tDone!")

                    # Save everything
                    save_encodings(all_encs, enc_file)
                    log.info("Saved encodings for model %s to %s", model_name, enc_file)     
                
                else:
                    raise ValueError("Model %s not found!" % model_name)
                        
            else:
                encsA, encsB, encsX, encsY = encs

            # run the test on the encodings
            log.info("Running test %s on %s", test, model_name)
            
            weat.run_test(encsA, encsB, encsX, encsY)


if __name__ == "__main__":
    main(sys.argv[1:])
