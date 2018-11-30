#!/usr/bin/env python3

import logging
import json
import string
import os


OUTPUT_PREFIX = 'sent-'

PROPER_NOUN_CONTEXTS = [
    'This is {}.',
    'That is {}.',
    'There is {}.',
    'Here is {}.',
    '{} is here.',
    '{} is there.',
]

ADJECTIVE_CONTEXTS = [
    'This is {}.',
    'That is {}.',
]

MASS_NOUN_CONTEXTS = [
    'This is {}.',
    'That is {}.',
    'There is {}.',
    'Here is {}.',
]

VERB_CONTEXTS = [
    'This will {}.',
    'This did {}.',
    'This can {}.',
    'This may {}.',
    'That will {}.',
    'That did {}.',
    'That can {}.',
    'That may {}.',
]

SINGULAR_VOWEL_CONTEXTS = [
    'This is an {}.',
    'That is an {}.',
    'There is an {}.',
    'Here is an {}.',
    'The {} is here.',
    'The {} is there.',
]

SINGULAR_CONSONANT_CONTEXTS = [
    'This is a {}.',
    'That is a {}.',
    'There is a {}.',
    'Here is a {}.',
    'The {} is here.',
    'The {} is there.',
]

PLURAL_CONTEXTS = [
    'These are {}.',
    'Those are {}.',
    'The {} are here.',
    'The {} are there.',
]

'A {} is a thing.'
'{} are things.'


def pluralize(s):
    if s.startswith('woman') or s.endswith('women'):
        return s.replace('woman', 'women')
    elif s.endswith('y'):
        return s[:-1] + 'ies'
    elif s.endswith('ch'):
        return s + 'es'
    elif s.endswith('sh'):
        return s + 'es'
    elif s.endswith('s'):
        return s + 'es'
    else:
        return s + 's'


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'Read word-level tests and generate corresponding sentence-level '
        'tests next to them using simple contexts.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_paths', nargs='+', metavar='input_path',
                        help='Paths to word-level json test files.  Output '
                             'files will be named by prepending {} to each '
                             'input filename.'.format(OUTPUT_PREFIX))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for input_path in args.input_paths:
        logging.info('Loading word-level test from {}'.format(input_path))
        with open(input_path) as f:
            sets = json.load(f)

        for (set_type, set_dict) in sets.items():
            sentences = []
            for example in set_dict['examples']:
                if example.startswith(string.ascii_uppercase):
                    sentences += [
                        context.format(example)
                        for context in PROPER_NOUN_CONTEXTS
                    ]
                elif example in (
                        'freedom',
                        'health',
                        'love',
                        'peace',
                        'cheer',
                        'heaven',
                        'loyalty',
                        'pleasure',
                        'laughter',
                        'filth',
                        'grief',
                        'hatred',
                        'poverty',
                        'agony',
                        'dynamite',
                        'teargas'):
                    sentences += [
                        context.format(example)
                        for context in MASS_NOUN_CONTEXTS
                    ]
                elif example in (
                        'loyal',
                        'honest',
                        'gentle',
                        'lucky',
                        'happy',
                        'ugly',
                        'rotten',
                        "soft",
                        "quiet",
                        "compromising",
                        "rational",
                        "calm",
                        "kind",
                        "agreeable",
                        "servile",
                        "pleasant",
                        "cautious",
                        "friendly",
                        "supportive",
                        "nice",
                        "mild",
                        "demure",
                        "passive",
                        "indifferent",
                        "submissive",
                        "shrill",
                        "loud",
                        "argumentative",
                        "irrational",
                        "angry",
                        "abusive",
                        "obnoxious",
                        "controlling",
                        "nagging",
                        "brash",
                        "hostile",
                        "emasculating",
                        "mean",
                        "harsh",
                        "sassy",
                        "aggressive",
                        "opinionated",
                        "domineering"):
                    sentences += [
                        context.format(example)
                        for context in ADJECTIVE_CONTEXTS
                    ]
                elif example in ('pollute', 'kill'):
                    sentences += [
                        context.format(example)
                        for context in VERB_CONTEXTS
                    ]
                else:
                    if any(example.startswith(v) for v in 'aeiou'):
                        sentences += [
                            context.format(example)
                            for context in SINGULAR_VOWEL_CONTEXTS
                        ]
                    else:
                        sentences += [
                            context.format(example)
                            for context in SINGULAR_CONSONANT_CONTEXTS
                        ]
                    sentences += [
                        context.format(pluralize(example))
                        for context in PLURAL_CONTEXTS
                    ]
            set_dict['examples'] = sentences

        (dirname, basename) = os.path.split(input_path)
        output_path = os.path.join(dirname, OUTPUT_PREFIX + basename)

        logging.info('Writing sentence-level test to {}'.format(output_path))
        with open(output_path, 'w') as f:
            json.dump(sets, f, indent=2)


if __name__ == '__main__':
    main()
