''' Main script for loading models and running WEAT tests '''
import sys
import argparse
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

import glove
import weat

TESTS = ["weat1", "weat2", "weat3", "weat4"]
MODELS = ["glove", "elmo"]

def handle_arguments(arguments):
    ''' Helper function for handling argument parsing '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--log_file', '-l', type=str, help="File to log to")

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

def main(arguments):
    ''' Main logic: parse args for tests to run and which models to evaluate '''
    args = handle_arguments(arguments)
    log.getLogger().addHandler(log.FileHandler(args.log_file))
    log.info("Parsed args: \n%s", args)

    #for wt in weat_tests:
    #  glove.create_weat_vec_files("../tests/"+wt+".txt")
    tests = split_comma_and_check(args.tests, TESTS, "test")
    models = split_comma_and_check(args.models, MODELS, "model")

    for test in tests:
        for model in models:
            log.info("Running test %s on %s", test, model)
            if "model" == "elmo":
                A, B, X, Y = weat.load_elmo_weat_test(test)
            else:
                A, B, X, Y = weat.load_weat_test(test)
            weat.run_test(A,B,X,Y)

if __name__ == "__main__":
    main(sys.argv[1:])
