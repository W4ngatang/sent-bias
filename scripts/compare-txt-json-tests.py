#!/usr/bin/env python3

import os
import json

TESTS_DIR = '../tests'

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info('Note: this script should be called from the "scripts" directory')

    for json_filename in os.listdir(TESTS_DIR):
        if not json_filename.startswith('.') and \
                json_filename.endswith('.jsonl'):
            json_path = os.path.join(TESTS_DIR, json_filename)
            with open(json_path) as f:
                j = json.load(f)

            txt_filename = json_filename[:json_filename.rindex('.')] + '.txt'
            txt_path = os.path.join(TESTS_DIR, txt_filename)
            t = dict()
            with open(txt_path) as f:
                for line in f:
                    set_type, set_name, *examples = line.rstrip().split('\t')
                    t[set_type] = dict(
                        category=set_name,
                        examples=examples,
                    )

            print('{:<40}: {}'.format(json_filename, j == t))
