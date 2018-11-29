#!/usr/bin/env python3


import os
import json


TEST_EXT = '.jsonl'
TESTS_DIR = '../tests'
ELMO_DIR = '../elmo'


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'Convert WEAT test data to ELMo-friendly format.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'test_name',
        help='Name of WEAT test to convert (for example, weat1).')
    args = parser.parse_args()

    with open(os.path.join(TESTS_DIR, args.test_name + TEST_EXT)) as f:
        test_data = json.load(f)

    if not os.path.isdir(ELMO_DIR):
        os.makedirs(ELMO_DIR)

    for (set_name, set_var) in (
            ('targ1', 'X'),
            ('targ2', 'Y'),
            ('attr1', 'A'),
            ('attr2', 'B')):
        elmo_path = os.path.join(
            ELMO_DIR,
            '{}.{}.txt'.format(args.test_name, set_var))
        with open(elmo_path, 'w') as f:
            for example in test_data[set_name]['examples']:
                f.write('{}\n'.format(example))
