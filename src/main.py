''' Main script for loading models and running WEAT tests '''
import os
import sys
import random
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import h5py
import numpy as np

from data import load_json, \
                 load_encodings, save_encodings, load_jiant_encodings
import weat
import encoders.glove as glove
import encoders.bow as bow
import encoders.infersent as infersent
import encoders.gensen as gensen
import encoders.elmo as elmo
import encoders.bert as bert
import tensorflow as tf
import tensorflow_hub as hub

TESTS = ['angry_black_woman_stereotype_b', 'angry_black_woman_stereotype',
         'heilman_double_bind_ambiguous_1+3-', 'heilman_double_bind_ambiguous_1-',
         'heilman_double_bind_ambiguous_1', 'heilman_double_bind_clear_1+3-',
         'heilman_double_bind_clear_1-', 'heilman_double_bind_clear_1',
         'project_implicit_arab-muslim', 'project_implicit_disability',
         'project_implicit_native', 'project_implicit_religion',
         'project_implicit_sexuality', 'project_implicit_skin-tone',
         'project_implicit_weapons', 'project_implicit_weight',
         'sent-weat1', 'sent-weat2', 'sent-weat3', 'sent-weat4',
         'weat1', 'weat2', 'weat3b', 'weat3', 'weat4',
         'weat5b', 'weat5', 'weat6b', 'weat6',
         'weat7b', 'weat7', 'weat8b', 'weat8',
         'weat9', 'weat10']
TEST_EXT = 'jsonl'
MODELS = ["glove", "infersent", "elmo", "gensen", "bow", "guse",
          "bert", "cove", "openai"]


def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(
        description='Run specified SEAT tests on specified models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tests', '-t', type=str, required=True,
                        help="WEAT tests to run (a comma-separated list; test files should be in `data_dir` and "
                             "have corresponding names, with extension {})".format(TEST_EXT))
    parser.add_argument('--models', '-m', type=str, required=True,
                        help="Models to evaluate (a comma-separated list; options: {})".format(','.join(MODELS)))
    parser.add_argument('--seed', '-s', type=int, help="Random seed", default=1111)
    parser.add_argument('--log_file', '-l', type=str,
                        help="File to log to")
    parser.add_argument('--glove_path', '-g', type=str,
                        help="File to GloVe vectors. Required if glove model is specified.")
    parser.add_argument('--ignore_cached_encs', '-i', action='store_true',
                        help="If set, ignore existing encodings and encode from scratch.")
    parser.add_argument('--dont_cache_encs', action='store_true',
                        help="If set, don't cache encodings to disk.")

    parser.add_argument('--data_dir', '-d', type=str,
                        help="Directory containing examples for each test",
                        default='tests')
    parser.add_argument('--exp_dir', type=str,
                        help="Directory from which to load and save vectors. "
                             "Files should be stored as h5py files.",
                        default='output')
    parser.add_argument('--n_samples', type=int,
                        help="Number of permutation test samples used when estimate p-values (exact test is used if "
                             "there are fewer than this many permutations)",
                        default=100000)

    parser.add_argument('--combine_method', type=str, choices=["max", "mean", "last", "concat"],
                        default="max", help="How to combine vector sequences")
    parser.add_argument('--infersent_dir', type=str,
                        help="Directory containing model files. Required if infersent model is specified.")
    parser.add_argument('--gensen_dir', type=str,
                        help="Directory containing model files. Required if gensen model is specified.")
    parser.add_argument('--gensen_version', type=str,
                        choices=["nli_large_bothskip", "nli_large_bothskip_parse", "nli_large_bothskip_2layer"],
                        default="nli_large_bothskip_parse", help="Version of gensen to use.")
    parser.add_argument('--cove_encs', type=str,
                        help="Directory containing precomputed CoVe encodings. Required if cove model is specified.")
    parser.add_argument('--elmo_combine', type=str, choices=["add", "concat"],
                        help="TODO", default="add")
    parser.add_argument('--openai_encs', type=str,
                        help="Directory containing precomputed OpenAI encodings. "
                             "Required if openai model is specified.")
    parser.add_argument('--bert_version', type=str, choices=["base", "large"],
                        help="Version of BERT to use.", default="base")
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
    log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

    args = handle_arguments(arguments)
    seed = random.randint(1, 100000) if args.seed < 0 else args.seed
    random.seed(seed)
    np.random.seed(seed)
    maybe_make_dir(args.exp_dir)
    if args.log_file:
        log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    tests = split_comma_and_check(
        args.tests,
        [
            entry[:-(len(TEST_EXT) + 1)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith('.') and entry.endswith(TEST_EXT)
        ],
        "test")
    models = split_comma_and_check(args.models, MODELS, "model")
    results = []
    for model_name in models:
        # Different models have different interfaces for things, but generally want to:
        # - if saved vectors aren't there:
        #    - load the model
        #    - load the test data
        #    - encode the vectors
        #    - dump the files into some storage
        # - else load the saved vectors '''
        log.info('Running tests for model {}'.format(model_name))

        model = None

        for test in tests:
            log.info('Running test {} for model {}'.format(test, model_name))
            enc_file = os.path.join(args.exp_dir, "%s.%s.h5" % (model_name, test))
            if not args.ignore_cached_encs and os.path.isfile(enc_file):
                log.info("Loading encodings from %s", enc_file)
                encs = load_encodings(enc_file)
                encs_targ1 = encs['targ1']
                encs_targ2 = encs['targ2']
                encs_attr1 = encs['attr1']
                encs_attr2 = encs['attr2']
            else:
                # load the test data
                encs = load_json(os.path.join(args.data_dir, "%s.%s" % (test, TEST_EXT)),
                                  split_sentence_into_list=bool(model == "bert"))

                # load the model and do model-specific encoding procedure
                log.info('Computing sentence encodings')
                if model_name == 'glove':
                    log.warn("GloVe is deprecating; use 'bow' instead!")
                    encs_targ1, encs_targ2, encs_attr1, encs_attr2 = weat.load_weat_test(test, path=args.data_dir)

                elif model_name == 'bow':
                    encs_targ1 = bow.encode(encs["targ1"]["examples"], args.glove_path, tokenize=True)
                    encs_targ2 = bow.encode(encs["targ2"]["examples"], args.glove_path, tokenize=True)
                    encs_attr1 = bow.encode(encs["attr1"]["examples"], args.glove_path, tokenize=True)
                    encs_attr2 = bow.encode(encs["attr2"]["examples"], args.glove_path, tokenize=True)

                elif model_name == 'infersent':
                    if model is None:
                        model = infersent.load_infersent(args.infersent_dir, args.glove_path, train_data='all')
                    model.build_vocab([s for s in encs["targ1"]["examples"] + encs["targ2"]["examples"] + encs["attr1"]["examples"] + encs["attr2"]["examples"]], tokenize=False)
                    log.info("Encoding sentences for test %s with model %s...", test, model_name)
                    encs_targ1 = infersent.encode(model, encs["targ1"]["examples"])
                    encs_targ2 = infersent.encode(model, encs["targ2"]["examples"])
                    encs_attr1 = infersent.encode(model, encs["attr1"]["examples"])
                    encs_attr2 = infersent.encode(model, encs["attr2"]["examples"])

                elif model_name =='gensen':
                    if model is None:
                        model = gensen.GenSenSingle(model_folder=os.path.join(args.gensen_dir, 'models'),
                                                    filename_prefix=args.gensen_version,
                                                    pretrained_emb=os.path.join(args.glove_path, 'glove.840B.300d.h5'))

                    encs_targ1 = gensen.encode(model, encs["targ1"]["examples"])
                    encs_targ2 = gensen.encode(model, encs["targ2"]["examples"])
                    encs_attr1 = gensen.encode(model, encs["attr1"]["examples"])
                    encs_attr2 = gensen.encode(model, encs["attr2"]["examples"])

                elif model_name =='guse':
                    enc = [[] * 512 for _ in range(4)]
                    encs_targ1, encs_targ2, encs_attr1, encs_attr2 = {}, {}, {}, {}

                    model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
                    os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
                    config = tf.ConfigProto()
                    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximum alloc gpu50% of MEM
                    config.gpu_options.allow_growth = True #allocate dynamically

                    # TODO(Alex): I don't think this is compatible with dictionaries
                    for i, sent in enumerate(encs): # iterate through the four word sets
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
                    encs = load_jiant_encodings(load_encs_from, n_header=1)

                elif model_name == 'elmo':
                    #encs_attr11, encs_attr21, encs_targ11, encs_targ21 = weat.load_elmo_weat_test(test, path='encodings/elmo/')
                    encs_targ1 = elmo.encode(encs["targ1"]["examples"], args.combine_method, args.elmo_combine)
                    encs_targ2 = elmo.encode(encs["targ2"]["examples"], args.combine_method, args.elmo_combine)
                    encs_attr1 = elmo.encode(encs["attr1"]["examples"], args.combine_method, args.elmo_combine)
                    encs_attr2 = elmo.encode(encs["attr2"]["examples"], args.combine_method, args.elmo_combine)

                elif model_name == "bert":
                    if args.bert_version == "large":
                        model, tokenizer = bert.load_model('bert-large-uncased')
                    else:
                        model, tokenizer = bert.load_model('bert-base-uncased')
                    encs_targ1 = bert.encode(model, tokenizer, encs["targ1"]["examples"], args.combine_method)
                    encs_targ2 = bert.encode(model, tokenizer, encs["targ2"]["examples"], args.combine_method)
                    encs_attr1 = bert.encode(model, tokenizer, encs["attr1"]["examples"], args.combine_method)
                    encs_attr2 = bert.encode(model, tokenizer, encs["attr2"]["examples"], args.combine_method)


                elif model_name == "openai":
                    load_encs_from = os.path.join(args.openai_encs, "%s.encs" % test)
                    encs = load_jiant_encodings(load_encs_from, n_header=1, is_openai=True)

                else:
                    raise ValueError("Model %s not found!" % model_name)

                encs["targ1"]["encs"] = encs_targ1
                encs["targ2"]["encs"] = encs_targ2
                encs["attr1"]["encs"] = encs_attr1
                encs["attr2"]["encs"] = encs_attr2

                log.info("\tDone!")
                if not args.dont_cache_encs:
                    log.info("Saving encodings to %s", enc_file)
                    save_encodings(encs, enc_file)

            enc = [e for e in encs["targ1"]['encs'].values()][0]
            d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)

            # run the test on the encodings
            log.info("Running SEAT...")
            log.info("Representation dimension: {}".format(d_rep))
            esize, pval = weat.run_test(encs, n_samples=args.n_samples)
            results.append((test, pval, esize))
            log.info("\n")

        log.info("Model: %s", model_name)
        for test, pval, esize in results:
            log.info("\tTest %s:\tp-val: %.5f\tesize: %.5f", test, pval, esize)


if __name__ == "__main__":
    main(sys.argv[1:])
