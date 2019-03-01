''' Main script for loading models and running WEAT tests '''

import os
import sys
import random
import re
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)  # noqa

from csv import DictWriter
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from data import (
    load_json, load_encodings, save_encodings, load_jiant_encodings,
)
import weat
import encoders.bow as bow
import encoders.infersent as infersent
import encoders.gensen as gensen
import encoders.elmo as elmo
import encoders.bert as bert


class ModelName(Enum):
    INFERSENT = 'infersent'
    ELMO = 'elmo'
    GENSEN = 'gensen'
    BOW = 'bow'
    GUSE = 'guse'
    BERT = 'bert'
    COVE = 'cove'
    OPENAI = 'openai'

TEST_EXT = '.jsonl'
MODEL_NAMES = [m.value for m in ModelName]
GENSEN_VERSIONS = ["nli_large_bothskip", "nli_large_bothskip_parse", "nli_large_bothskip_2layer", "nli_large"]
BERT_VERSIONS = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased"]


def test_sort_key(test):
    '''
    Return tuple to be used as a sort key for the specified test name.
   Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key


def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(
        description='Run specified SEAT tests on specified models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tests', '-t', type=str,
                        help="WEAT tests to run (a comma-separated list; test files should be in `data_dir` and "
                             "have corresponding names, with extension {}). Default: all tests.".format(TEST_EXT))
    parser.add_argument('--models', '-m', type=str,
                        help="Models to evaluate (a comma-separated list; options: {}). "
                             "Default: all models.".format(','.join(MODEL_NAMES)))
    parser.add_argument('--seed', '-s', type=int, help="Random seed", default=1111)
    parser.add_argument('--log_file', '-l', type=str,
                        help="File to log to")
    parser.add_argument('--results_path', type=str,
                        help="Path where TSV results file will be written")
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
    parser.add_argument('--parametric', action='store_true',
                        help='Use parametric test (normal assumption) to compute p-values.')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Use CPU to encode sentences.')
    parser.add_argument('--glove_path', '-g', type=str,
                        help="File to GloVe vectors in .txt format. "
                             "Required if bow or infersent models are specified.")

    elmo_group = parser.add_argument_group(ModelName.ELMO.value, 'Options for ELMo model')
    elmo_group.add_argument('--time_combine_method', type=str, choices=["max", "mean", "concat", "last"],
                            help="How to combine word representations in ELMo", default="mean")
    elmo_group.add_argument('--layer_combine_method', type=str, choices=["add", "mean", "concat", "last"],
                            help="How to combine layers in ELMo", default="add")

    infersent_group = parser.add_argument_group(ModelName.INFERSENT.value, 'Options for InferSent model')
    infersent_group.add_argument('--infersent_dir', type=str,
                                 help="Directory containing model files. Required if infersent model is specified.")

    gensen_group = parser.add_argument_group(ModelName.GENSEN.value, 'Options for GenSen model')
    gensen_group.add_argument('--glove_h5_path', type=str,
                              help="File to GloVe vectors in .h5 (HDF5) format.")
    gensen_group.add_argument('--gensen_dir', type=str,
                              help="Directory containing model files. Required if gensen model is specified.")
    gensen_group.add_argument('--gensen_version', type=str,
                              help="Version of gensen to use.  Two versions may be passed, separated by commas, in "
                                   "which case the respective models will be concatenated.  "
                                   "Options: {}".format(','.join(GENSEN_VERSIONS)),
                              default="nli_large_bothskip_parse,nli_large_bothskip")

    cove_group = parser.add_argument_group(ModelName.COVE.value, 'Options for CoVe model')
    cove_group.add_argument('--cove_encs', type=str,
                            help="Directory containing precomputed CoVe encodings. "
                                 "Required if cove model is specified.")

    openai_group = parser.add_argument_group(ModelName.OPENAI.value, 'Options for OpenAI model')
    openai_group.add_argument('--openai_encs', type=str,
                              help="Directory containing precomputed OpenAI encodings. "
                                   "Required if openai model is specified.")

    bert_group = parser.add_argument_group(ModelName.BERT.value, 'Options for BERT model')
    bert_group.add_argument('--bert_version', type=str, choices=BERT_VERSIONS,
                            help="Version of BERT to use.", default="bert-large-cased")

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
    if args.seed >= 0:
        log.info('Seeding random number generators with {}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
    maybe_make_dir(args.exp_dir)
    if args.log_file:
        log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    all_tests = sorted(
        [
            entry[:-len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith('.') and entry.endswith(TEST_EXT)
        ],
        key=test_sort_key
    )
    log.debug('Tests found:')
    for test in all_tests:
        log.debug('\t{}'.format(test))

    tests = split_comma_and_check(args.tests, all_tests, "test") if args.tests is not None else all_tests
    log.info('Tests selected:')
    for test in tests:
        log.info('\t{}'.format(test))

    models = split_comma_and_check(args.models, MODEL_NAMES, "model") if args.models is not None else MODEL_NAMES
    log.info('Models selected:')
    for model in models:
        log.info('\t{}'.format(model))


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

        if model_name == ModelName.BOW.value:
            model_options = ''
            if args.glove_path is None:
                raise Exception('glove_path must be specified for {} model'.format(model_name))
        elif model_name == ModelName.INFERSENT.value:
            if args.glove_path is None:
                raise Exception('glove_path must be specified for {} model'.format(model_name))
            if args.infersent_dir is None:
                raise Exception('infersent_dir must be specified for {} model'.format(model_name))
            model_options = ''
        elif model_name == ModelName.GENSEN.value:
            if args.glove_h5_path is None:
                raise Exception('glove_h5_path must be specified for {} model'.format(model_name))
            if args.gensen_dir is None:
                raise Exception('gensen_dir must be specified for {} model'.format(model_name))
            gensen_version_list = split_comma_and_check(args.gensen_version, GENSEN_VERSIONS, "gensen_prefix")
            if len(gensen_version_list) > 2:
                raise ValueError('gensen_version can only have one or two elements')
            model_options = 'version=' + args.gensen_version
        elif model_name == ModelName.GUSE.value:
            model_options = ''
        elif model_name == ModelName.COVE.value:
            if args.cove_encs is None:
                raise Exception('cove_encs must be specified for {} model'.format(model_name))
            model_options = ''
        elif model_name == ModelName.ELMO.value:
            model_options = 'time_combine={};layer_combine={}'.format(
                args.time_combine_method, args.layer_combine_method)
        elif model_name == ModelName.BERT.value:
            model_options = 'version=' + args.bert_version
        elif model_name == ModelName.OPENAI.value:
            if args.openai_encs is None:
                raise Exception('openai_encs must be specified for {} model'.format(model_name))
            model_options = ''
        else:
            raise ValueError("Model %s not found!" % model_name)

        model = None

        for test in tests:
            log.info('Running test {} for model {}'.format(test, model_name))
            enc_file = os.path.join(args.exp_dir, "%s.%s.h5" % (
                "%s;%s" % (model_name, model_options) if model_options else model_name,
                test))
            if not args.ignore_cached_encs and os.path.isfile(enc_file):
                log.info("Loading encodings from %s", enc_file)
                encs = load_encodings(enc_file)
                encs_targ1 = encs['targ1']
                encs_targ2 = encs['targ2']
                encs_attr1 = encs['attr1']
                encs_attr2 = encs['attr2']
            else:
                # load the test data
                encs = load_json(os.path.join(args.data_dir, "%s%s" % (test, TEST_EXT)))

                # load the model and do model-specific encoding procedure
                log.info('Computing sentence encodings')
                if model_name == ModelName.BOW.value:
                    encs_targ1 = bow.encode(encs["targ1"]["examples"], args.glove_path)
                    encs_targ2 = bow.encode(encs["targ2"]["examples"], args.glove_path)
                    encs_attr1 = bow.encode(encs["attr1"]["examples"], args.glove_path)
                    encs_attr2 = bow.encode(encs["attr2"]["examples"], args.glove_path)

                elif model_name == ModelName.INFERSENT.value:
                    if model is None:
                        model = infersent.load_infersent(args.infersent_dir, args.glove_path, train_data='all',
                                                         use_cpu=args.use_cpu)
                    model.build_vocab(
                        [
                            example
                            for k in ('targ1', 'targ2', 'attr1', 'attr2')
                            for example in encs[k]['examples']
                        ],
                        tokenize=True)
                    log.info("Encoding sentences for test %s with model %s...", test, model_name)
                    encs_targ1 = infersent.encode(model, encs["targ1"]["examples"])
                    encs_targ2 = infersent.encode(model, encs["targ2"]["examples"])
                    encs_attr1 = infersent.encode(model, encs["attr1"]["examples"])
                    encs_attr2 = infersent.encode(model, encs["attr2"]["examples"])

                elif model_name == ModelName.GENSEN.value:
                    if model is None:
                        gensen_1 = gensen.GenSenSingle(
                            model_folder=args.gensen_dir,
                            filename_prefix=gensen_version_list[0],
                            pretrained_emb=args.glove_h5_path,
                            cuda=not args.use_cpu)
                        model = gensen_1

                        if len(gensen_version_list) == 2:
                            gensen_2 = gensen.GenSenSingle(
                                model_folder=args.gensen_dir,
                                filename_prefix=gensen_version_list[1],
                                pretrained_emb=args.glove_h5_path,
                                cuda=not args.use_cpu)
                            model = gensen.GenSen(gensen_1, gensen_2)

                    vocab = gensen.build_vocab([
                        s
                        for set_name in ('targ1', 'targ2', 'attr1', 'attr2')
                        for s in encs[set_name]["examples"]
                    ])

                    model.vocab_expansion(vocab)

                    encs_targ1 = gensen.encode(model, encs["targ1"]["examples"])
                    encs_targ2 = gensen.encode(model, encs["targ2"]["examples"])
                    encs_attr1 = gensen.encode(model, encs["attr1"]["examples"])
                    encs_attr2 = gensen.encode(model, encs["attr2"]["examples"])

                elif model_name == ModelName.GUSE.value:
                    model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
                    if args.use_cpu:
                        kwargs = dict(device_count={'GPU': 0})
                    else:
                        kwargs = dict()
                    config = tf.ConfigProto(**kwargs)
                    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximum alloc gpu50% of MEM
                    config.gpu_options.allow_growth = True  # allocate dynamically
                    with tf.Session(config=config) as session:
                        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                        def guse_encode(sents):
                            encs_node = model(sents)
                            encs = session.run(encs_node)
                            encs_d = {sents[j]: enc for j, enc in enumerate(np.array(encs).tolist())}
                            return encs_d

                        encs_targ1 = guse_encode(encs["targ1"]["examples"])
                        encs_targ2 = guse_encode(encs["targ2"]["examples"])
                        encs_attr1 = guse_encode(encs["attr1"]["examples"])
                        encs_attr2 = guse_encode(encs["attr2"]["examples"])

                elif model_name == ModelName.COVE.value:
                    load_encs_from = os.path.join(args.cove_encs, "%s.encs" % test)
                    encs = load_jiant_encodings(load_encs_from, n_header=1)

                elif model_name == ModelName.ELMO.value:
                    kwargs = dict(time_combine_method=args.time_combine_method,
                                  layer_combine_method=args.layer_combine_method)
                    encs_targ1 = elmo.encode(encs["targ1"]["examples"], **kwargs)
                    encs_targ2 = elmo.encode(encs["targ2"]["examples"], **kwargs)
                    encs_attr1 = elmo.encode(encs["attr1"]["examples"], **kwargs)
                    encs_attr2 = elmo.encode(encs["attr2"]["examples"], **kwargs)

                elif model_name == ModelName.BERT.value:
                    model, tokenizer = bert.load_model(args.bert_version)
                    encs_targ1 = bert.encode(model, tokenizer, encs["targ1"]["examples"])
                    encs_targ2 = bert.encode(model, tokenizer, encs["targ2"]["examples"])
                    encs_attr1 = bert.encode(model, tokenizer, encs["attr1"]["examples"])
                    encs_attr2 = bert.encode(model, tokenizer, encs["attr2"]["examples"])

                elif model_name == ModelName.OPENAI.value:
                    load_encs_from = os.path.join(args.openai_encs, "%s.encs" % test)
                    #encs = load_jiant_encodings(load_encs_from, n_header=1, is_openai=True)
                    encs = load_encodings(load_encs_from)
                    encs_targ1 = encs["targ1"]["encs"]
                    encs_targ2 = encs["targ2"]["encs"]
                    encs_attr1 = encs["attr1"]["encs"]
                    encs_attr2 = encs["attr2"]["encs"]

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
            esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
            results.append(dict(
                model=model_name,
                options=model_options,
                test=test,
                p_value=pval,
                effect_size=esize,
                num_targ1=len(encs['targ1']['encs']),
                num_targ2=len(encs['targ2']['encs']),
                num_attr1=len(encs['attr1']['encs']),
                num_attr2=len(encs['attr2']['encs'])))

        log.info("Model: %s", model_name)
        log.info('Options: {}'.format(model_options))
        for r in results:
            log.info("\tTest {test}:\tp-val: {p_value:.9f}\tesize: {effect_size:.2f}".format(**r))

    if args.results_path is not None:
        log.info('Writing results to {}'.format(args.results_path))
        with open(args.results_path, 'w') as f:
            writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            for r in results:
                writer.writerow(r)


if __name__ == "__main__":
    main(sys.argv[1:])
