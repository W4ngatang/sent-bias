''' Main script for loading models and running WEAT tests '''
import os
import sys
import random
import argparse
import logging as log
import h5py # pylint: disable=import-error
import numpy as np
from data import load_sents, load_encodings, save_encodings, \
                 load_jiant_encodings
import weat
import encoders.glove as glove
import encoders.bow as bow
import encoders.infersent as infersent
import encoders.gensen as gensen
import encoders.elmo as elmo
import encoders.bert as bert
import tensorflow as tf
import tensorflow_hub as hub
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

TESTS = ["weat1", "weat2", "weat3", "weat4",
         "sent-weat1", "sent-weat2", "sent-weat3", "sent-weat4"]
MODELS = ["glove", "infersent", "elmo", "gensen", "bow", "guse",
          "bert", "cove", "openai"]


def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', '-s', type=int, help="Random seed", default=1111)
    parser.add_argument('--log_file', '-l', type=str, help="File to log to")
    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test")
    parser.add_argument('--exp_dir', type=str,
                        help="Directory from which to load and save vectors. " +
                        "Files should be stored as h5py files.")
    parser.add_argument('--glove_path', '-g', type=str, help="File to GloVe vectors")
    parser.add_argument('--ignore_cached_encs', '-i', type=str,
                        help="1 if ignore existing encodings and encode from scratch")

    parser.add_argument('--tests', '-t', type=str, help="WEAT tests to run")
    parser.add_argument('--n_samples', type=int, help="Number of samples to estimate p-value", default=100000)

    parser.add_argument('--models', '-m', type=str, help="Model to evaluate")
    parser.add_argument('--combine_method', type=str, choices=["max", "mean", "last", "concat"],
                        default="max", help="How to combine vector sequences")
    parser.add_argument('--infersent_dir', type=str, help="Directory containing model files")
    parser.add_argument('--gensen_dir', type=str, help="Directory containing model files")
    parser.add_argument('--gensen_version', type=str,
                        choices=["nli_large_bothskip", "nli_large_bothskip_parse", "nli_large_bothskip_2layer"],
                        default="nli_large_bothskip_parse", help="Version of gensen to use.")
    parser.add_argument('--cove_encs', type=str, help="Directory containing precomputed CoVe encodings")
    parser.add_argument('--elmo_combine', type=str, choices=["add", "concat"],
                        default="add", help="Directory containing precomputed CoVe encodings")
    parser.add_argument('--openai_encs', type=str, help="Directory containing precomputed OpenAI encodings")
    parser.add_argument('--bert_version', type=str, choices=["base", "large"], help="Version of BERT to use.")
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


def main(arguments):
    ''' Main logic: parse args for tests to run and which models to evaluate '''
    args = handle_arguments(arguments)
    seed = random.randint(1, 100000) if args.seed < 0 else args.seed
    random.seed(seed)
    np.random.seed(seed)
    maybe_make_dir(args.exp_dir)
    log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    tests = split_comma_and_check(args.tests, TESTS, "test")
    models = split_comma_and_check(args.models, MODELS, "model")
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
                sents = load_sents(os.path.join(args.data_dir, "%s.txt" % test), split_sentence_into_list=bool(model=="bert"))
                assert len(sents) == 4
                assert isinstance(sents[0], list)

                # load the model and do model-specific encoding procedure
                if model_name == 'glove':
                    log.warn("GloVe is deprecating; use 'bow' instead!")
                    encs_targ1, encs_targ2, encs_attr1, encs_attr2 = weat.load_weat_test(test, path=args.data_dir)

                elif model_name == 'bow':
                    encs_targ1 = bow.encode(sents[0], args.glove_path)
                    encs_targ2 = bow.encode(sents[1], args.glove_path)
                    encs_attr1 = bow.encode(sents[2], args.glove_path)
                    encs_attr2 = bow.encode(sents[3], args.glove_path)

                elif model_name == 'infersent':
                    if model is None:
                        model = infersent.load_infersent(args.infersent_dir, args.glove_path, train_data='all')
                    model.build_vocab([s for s in sents[0] + sents[1] + sents[2] + sents[3]], tokenize=False)
                    log.info("Encoding sentences for test %s with model %s...", test, model_name)
                    encs_targ1 = infersent.encode(model, sents[0])
                    encs_targ2 = infersent.encode(model, sents[1])
                    encs_attr1 = infersent.encode(model, sents[2])
                    encs_attr2 = infersent.encode(model, sents[3])

                elif model_name =='gensen':
                    if model is None:
                        model = gensen.GenSenSingle(model_folder=os.path.join(args.gensen_dir, 'models'),
                                                    filename_prefix=args.gensen_version,
                                                    pretrained_emb=os.path.join(args.glove_path, 'glove.840B.300d.h5'))

                    encs_targ1 = gensen.encode(model, sents[0])
                    encs_targ2 = gensen.encode(model, sents[1])
                    encs_attr1 = gensen.encode(model, sents[2])
                    encs_attr2 = gensen.encode(model, sents[3])

                elif model_name =='guse':
                    enc = [[] * 512 for _ in range(4)]
                    encs_targ1, encs_targ2, encs_attr1, encs_attr2 = {}, {}, {}, {}

                    model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
                    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
                    config = tf.ConfigProto()
                    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximum alloc gpu50% of MEM
                    config.gpu_options.allow_growth = True #allocate dynamically

                    for i, sent in enumerate(sents): # iterate through the four word sets
                        embeddings = model(sent) # embed the word set
                        with tf.Session(config=config) as session:
                            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                            enc[i] = session.run(embeddings)
                            if i == 0:
                                for j, embedding in enumerate(np.array(enc[i]).tolist()):
                                    encs_targ1[sent[j]] = embedding
                            elif i == 1:
                                for j, embedding in enumerate(np.array(enc[i]).tolist()):
                                    encs_targ2[sent[j]] = embedding
                            elif i == 2:
                                for j, embedding in enumerate(np.array(enc[i]).tolist()):
                                    encs_attr1[sent[j]] = embedding
                            elif i == 3:
                                for j, embedding in enumerate(np.array(enc[i]).tolist()):
                                    encs_attr2[sent[j]] = embedding

                elif model_name == "cove":
                    load_encs_from = os.path.join(args.cove_encs, "%s.encs" % test)
                    encs_targ1, encs_targ2, encs_attr1, encs_attr2 = load_jiant_encodings(load_encs_from, n_header=1)

                elif model_name == 'elmo':
                    #encs_attr11, encs_attr21, encs_targ11, encs_targ21 = weat.load_elmo_weat_test(test, path='encodings/elmo/')
                    encs_targ1 = elmo.encode(sents[0], args.combine_method, args.elmo_combine)
                    encs_targ2 = elmo.encode(sents[1], args.combine_method, args.elmo_combine)
                    encs_attr1 = elmo.encode(sents[2], args.combine_method, args.elmo_combine)
                    encs_attr2 = elmo.encode(sents[3], args.combine_method, args.elmo_combine)

                elif model_name == "bert":
                    if args.bert_version == "large":
                        model, tokenizer = bert.load_model('bert-large-uncased')
                    else:
                        model, tokenizer = bert.load_model('bert-base-uncased')
                    encs_targ1 = bert.encode(model, tokenizer, sents[0], args.combine_method)
                    encs_targ2 = bert.encode(model, tokenizer, sents[1], args.combine_method)
                    encs_attr1 = bert.encode(model, tokenizer, sents[2], args.combine_method)
                    encs_attr2 = bert.encode(model, tokenizer, sents[3], args.combine_method)


                elif model_name == "openai":
                    load_encs_from = os.path.join(args.openai_encs, "%s.encs" % test)
                    encs_targ1, encs_targ2, encs_attr1, encs_attr2 = load_jiant_encodings(load_encs_from, n_header=1, is_openai=True)

                else:
                    raise ValueError("Model %s not found!" % model_name)

                #all_encs = [encs_targ1, encs_targ2, encs_attr1, encs_attr2]
                #save_encodings(all_encs, enc_file)
                log.info("\tDone!")
                log.info("Saved encodings for model %s to %s", model_name, enc_file)
            else:
                encs_targ1, encs_targ2, encs_attr1, encs_attr2 = encs

            enc = [e for e in encs_targ1.values()][0]
            d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)

            # run the test on the encodings
            log.info("Running test %s on %s", test, model_name)
            log.info("Representation dimension: %d", d_rep)
            weat.run_test(encs_targ1, encs_targ2, encs_attr1, encs_attr2, args.n_samples)


if __name__ == "__main__":
    main(sys.argv[1:])
