#!/usr/bin/env python3

import logging
import json
import string
import os
import re


WOMAN_RE = re.compile(r'\b(?:woman)\b')
MAN_RE = re.compile(r'\b(?:man)\b')
PERSON_RE = re.compile(r'\b(?:woman|man|female|male)\b')

OUTPUT_PREFIX = 'sent-'

ADJECTIVES = {
    'loyal',
    'honest',
    'gentle',
    'lucky',
    'happy',
    'ugly',
    'rotten',
    'soft',
    'quiet',
    'compromising',
    'rational',
    'calm',
    'kind',
    'agreeable',
    'servile',
    'pleasant',
    'cautious',
    'friendly',
    'supportive',
    'nice',
    'mild',
    'demure',
    'passive',
    'indifferent',
    'submissive',
    'shrill',
    'loud',
    'argumentative',
    'irrational',
    'angry',
    'abusive',
    'obnoxious',
    'controlling',
    'nagging',
    'brash',
    'hostile',
    'emasculating',
    'mean',
    'harsh',
    'sassy',
    'aggressive',
    'opinionated',
    'domineering',
}

MASS_NOUNS = {
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
    'teargas',
}

VERBS = {
    'pollute',
    'kill',
}

NAME_TEMPLATES = (
    'This is {term}.',
    'That is {term}.',
    'There is {term}.',
    'Here is {term}.',
    '{term} is here.',
    '{term} is there.',
    '{term} is a person.',
    'The person\'s name is {term}.',
)

ADJECTIVE_TEMPLATES = (
    'This is {term}.',
    'That is {term}.',
    'They are {term}.',
)

MASS_NOUN_TEMPLATES = (
    'This is {term}.',
    'That is {term}.',
    'There is {term}.',
    'Here is {term}.',
    'It is {term}.',
)

VERB_TEMPLATES = (
    'This will {term}.',
    'This did {term}.',
    'This can {term}.',
    'This may {term}.',
    'That will {term}.',
    'That did {term}.',
    'That can {term}.',
    'That may {term}.',
)

SINGULAR_NOUN_TEMPLATES = (
    'This is {article} {term}.',
    'That is {article} {term}.',
    'There is {article} {term}.',
    'Here is {article} {term}.',
    'The {term} is here.',
    'The {term} is there.',
)

PLURAL_NOUN_TEMPLATES = (
    'These are {term}.',
    'Those are {term}.',
    'They are {term}.',
    'The {term} are here.',
    'The {term} are there.',
)

SINGULAR_PERSON_TEMPLATES = (
    'A {term} is a person.',
)

PLURAL_PERSON_TEMPLATES = (
    '{term} are people.',
)

SINGULAR_THING_TEMPLATES = (
    'A {term} is a thing.',
    'It is a {term}.',
)

PLURAL_THING_TEMPLATES = (
    '{term} are things.',
)


def fill_template(template, term):
    article = 'an' if any(term.startswith(c) for c in 'aeiouAEIOU') else 'a'
    sentence = template.format(article=article, term=term)
    return sentence[0].upper() + sentence[1:]


def pluralize(s):
    if WOMAN_RE.search(s) is not None:
        return WOMAN_RE.sub('women', s)
    elif MAN_RE.search(s) is not None:
        return MAN_RE.sub('men', s)
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
        'tests next to them using simple sentence templates.',
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
            for term in set_dict['examples']:
                if any(term.startswith(c) for c in string.ascii_uppercase):
                    sentences += [
                        fill_template(template, term)
                        for template in NAME_TEMPLATES
                    ]
                elif term in ADJECTIVES:
                    sentences += [
                        fill_template(template, term)
                        for template in ADJECTIVE_TEMPLATES
                    ]
                elif term in VERBS:
                    sentences += [
                        fill_template(template, term)
                        for template in VERB_TEMPLATES
                    ]
                elif term in MASS_NOUNS:
                    sentences += [
                        fill_template(template, term)
                        for template in MASS_NOUN_TEMPLATES
                    ]
                else:
                    sentences += [
                        fill_template(template, term)
                        for template in SINGULAR_NOUN_TEMPLATES + (
                            SINGULAR_PERSON_TEMPLATES
                            if PERSON_RE.search(term) is not None
                            else SINGULAR_THING_TEMPLATES
                        )
                    ]
                    sentences += [
                        fill_template(template, pluralize(term))
                        for template in PLURAL_NOUN_TEMPLATES + (
                            PLURAL_PERSON_TEMPLATES
                            if PERSON_RE.search(term) is not None
                            else PLURAL_THING_TEMPLATES
                        )
                    ]

            set_dict['examples'] = sentences

        (dirname, basename) = os.path.split(input_path)
        output_path = os.path.join(dirname, OUTPUT_PREFIX + basename)

        logging.info('Writing sentence-level test to {}'.format(output_path))
        with open(output_path, 'w') as f:
            json.dump(sets, f, indent=2)


if __name__ == '__main__':
    main()
