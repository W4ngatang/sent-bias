#!/usr/bin/env python3

import logging
import json
import string
import os
import re
import random


WOMAN_RE = re.compile(r'\b(?:woman)\b')
MAN_RE = re.compile(r'\b(?:man)\b')
PERSON_RE = re.compile(
    r'\b(?:woman|man|female|male|girl|boy|sister|brother|daughter|son|'
    r'mother|father|aunt|uncle|grandmother|grandfather|American)\b')

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
    'wonderful',
    'terrible',
    'horrible',
    'nasty',
    'awful',
    'professional',
    'evil',
    'hopeless',
    'sad',
    'gloomy',
    'tearful',
    'miserable',
    'depressed',
    'sick',
    'impermanent',
    'unstable',
    'variable',
    'fleeting',
    'short-term',
    'brief',
    'occasional',
    'stable',
    'constant',
    'persistent',
    'chronic',
    'prolonged',
    'competent',
    'productive',
    'effective',
    'ambitious',
    'active',
    'decisive',
    'strong',
    'tough',
    'bold',
    'assertive',
    'incompetent',
    'unproductive',
    'ineffective',
    'unambitious',
    'passive',
    'indecisive',
    'weak',
    'gentle',
    'timid',
    'unassertive',
    'agreeable',
    'fair',
    'honest',
    'trustworthy',
    'selfless',
    'accommodating',
    'likable',
    'liked',
    'abrasive',
    'conniving',
    'manipulative',
    'dishonest',
    'selfish',
    'pushy',
    'unlikable',
    'unliked',
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
    'joy',
    'physics',
    'calculus',
    'science',
    'chemistry',
    'astronomy',
    'NASA',
    'poetry',
    'art',
    'literature',
    'math',
    'algebra',
    'geometry',
    'always',
    'forever',
    'vomit',
}

PLURAL_NOUNS = {
    'parents',
    'children',
    'cousins',
    'relatives',
    'numbers',
    'equations',
}

VERBS = {
    'pollute',
    'kill',
    'stink',
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

SUBJECT_PRONOUN_TEMPLATES = (
    '{term} is here.',
    '{term} is there.',
    'Here {term} is.',
    'There {term} is.',
    '{term} is a person.',
)

OBJECT_PRONOUN_TEMPLATES = (
    'It is {term}.',
    'This is {term}.',
    'That is {term}.',
)

POSSESSIVE_PRONOUN_TEMPLATES = (
    'This is {term}.',
    'That is {term}.',
    'There is {term}.',
    'Here is {term}.',
    'It is {term}.',
    '{term} is there.',
    '{term} is here.',
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
    '{article} {term} is a person.',
)

PLURAL_PERSON_TEMPLATES = (
    '{term} are people.',
)

SINGULAR_THING_TEMPLATES = (
    '{article} {term} is a thing.',
    'It is {article} {term}.',
)

PLURAL_THING_TEMPLATES = (
    '{term} are things.',
)


def fill_template(template, term):
    article = (
        'an'
        if (
            (
                term.startswith('honor') or any(
                    term.startswith(c) for c in 'aeiouAEIOU'
                )
            ) and not (
                term.startswith('European') or term.startswith('Ukrainian')
            )
        )
        else 'a'
    )
    sentence = template.format(article=article, term=term)
    return sentence[0].upper() + sentence[1:]


def singularize(s):
    if s == 'children':
        return 'child'
    elif s.endswith('s'):
        return s[:-1]
    else:
        return s


def pluralize(s):
    if WOMAN_RE.search(s) is not None:
        return WOMAN_RE.sub('women', s)
    elif MAN_RE.search(s) is not None:
        return MAN_RE.sub('men', s)
    elif s.endswith('y') and s[-2] not in 'aeiou':
        return s[:-1] + 'ies'
    elif s.endswith('ch'):
        return s + 'es'
    elif s.endswith('sh'):
        return s + 'es'
    elif s.endswith('s'):
        return s + 'es'
    else:
        return s + 's'


def truncate_lists(list1, list2):
    '''
    Truncate `list1`, `list2` to the minimum of their lengths by
    randomly removing items.
    '''
    min_len = min(len(list1), len(list2))
    list1 = [x for (i, x) in sorted(random.sample(list(enumerate(list1)), min_len))]
    list2 = [x for (i, x) in sorted(random.sample(list(enumerate(list2)), min_len))]
    return (list1, list2)


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
                if any(term.startswith(c) for c in string.ascii_uppercase) and \
                        not term.endswith('American') and \
                        term != term.upper():
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
                elif term in ('he', 'she'):
                    sentences += [
                        fill_template(template, term)
                        for template in SUBJECT_PRONOUN_TEMPLATES
                    ]
                elif term in ('him', 'her'):
                    sentences += [
                        fill_template(template, term)
                        for template in OBJECT_PRONOUN_TEMPLATES
                    ]
                elif term in ('his', 'hers'):
                    sentences += [
                        fill_template(template, term)
                        for template in POSSESSIVE_PRONOUN_TEMPLATES
                    ]
                else:
                    if term in PLURAL_NOUNS:
                        singular_term = singularize(term)
                        plural_term = term
                    else:
                        singular_term = term
                        plural_term = pluralize(term)
                    sentences += [
                        fill_template(template, singular_term)
                        for template in SINGULAR_NOUN_TEMPLATES + (
                            SINGULAR_PERSON_TEMPLATES
                            if PERSON_RE.search(term) is not None
                            else SINGULAR_THING_TEMPLATES
                        )
                    ]
                    sentences += [
                        fill_template(template, plural_term)
                        for template in PLURAL_NOUN_TEMPLATES + (
                            PLURAL_PERSON_TEMPLATES
                            if PERSON_RE.search(term) is not None
                            else PLURAL_THING_TEMPLATES
                        )
                    ]

            set_dict['examples'] = sentences

        if len(sets['targ1']['examples']) != len(sets['targ2']['examples']):
            logging.info(
                'Truncating targ1, targ2 to have same size (current sizes: {}, {})'.format(
                    len(sets['targ1']['examples']), len(sets['targ2']['examples'])))
            (sets['targ1']['examples'], sets['targ2']['examples']) = truncate_lists(
                sets['targ1']['examples'], sets['targ2']['examples'])

        (dirname, basename) = os.path.split(input_path)
        output_path = os.path.join(dirname, OUTPUT_PREFIX + basename)

        logging.info('Writing sentence-level test to {}'.format(output_path))
        with open(output_path, 'w') as f:
            json.dump(sets, f, indent=2)


if __name__ == '__main__':
    main()
