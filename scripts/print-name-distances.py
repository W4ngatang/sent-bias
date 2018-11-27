#!/usr/bin/env python3

import logging

import gensim
import numpy as np


NAME_SETS = {
    'Black female associated names': '''
Aisha
Ebony
Keisha
Kenya
Latonya
Lakisha
Latoya
Tamika
Imani
Shanice
Aaliyah
Precious
Nia
Deja
Diamond
Latanya
Latisha
'''.strip().split(),
    'White female associated names': '''
Allison
Anne
Carrie
Emily
Jill
Laurie
Kristen
Meredith
Molly
Amy
Claire
Katie
Madeline
Katelyn
Emma
'''.strip().split(),
    'Black male associated names': '''
Darnell
Hakim
Jermaine
Kareem
Jamal
Leroy
Rasheed
Tremayne
DeShawn
DeAndre
Marquis
Terrell
Malik
Trevon
Tyrone
'''.strip().split(),
    'White male associated names': '''
Brad
Brendan
Geoffrey
Greg
Brett
Jay
Matthew
Neil
Jake
Connor
Tanner
Wyatt
Cody
Dustin
Luke
Jack
'''.strip().split(),
}


def norm2(x):
    return np.sqrt(x.dot(x))


def mean_vector(vectors):
    return np.mean(np.vstack(tuple(vectors)), axis=0)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'Print distances from hardcoded sets of names to their local '
        '(within-set) and global (across-sets) means, under a specified '
        'word embedding model.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'model_path',
        help='Path to word embedding model in word2vec text format (loadable '
             'with `gensim.models.KeyedVectors.load_word2vec_format`; use '
             '`python -m gensim.scripts.glove2word2vec` to convert from GloVe '
             'format to word2vec format).',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logging.info('loading model from {}'.format(args.model_path))
    m = gensim.models.KeyedVectors.load_word2vec_format(args.model_path)

    logging.info('computing global centroid')
    all_vectors = []
    for (name_set_name, name_set) in NAME_SETS.items():
        for name in name_set:
            all_vectors.append(m[name])
    global_centroid = mean_vector(all_vectors)

    for (name_set_name, name_set) in NAME_SETS.items():
        logging.info('computing distances for {}'.format(name_set_name))

        name_vectors = dict()
        for name in name_set:
            try:
                name_vectors[name] = m[name]
            except KeyError:
                logging.warning('name not in vocabulary: {}'.format(name))

        logging.info(
            'computing outliers among {} vectors'.format(len(name_vectors)))

        local_centroid = mean_vector(name_vectors.values())

        for (centroid_name, centroid) in (('global', global_centroid),
                                          ('local', local_centroid)):
            name_distances = dict(
                (name, norm2(vector - centroid))
                for (name, vector)
                in name_vectors.items())

            print('{} ({})'.format(name_set_name, centroid_name))
            for (name, distance) in sorted(name_distances.items(),
                                           key=lambda p: p[1]):
                print('{:<20}: {:7.2f}'.format(name, distance))
            print()


if __name__ == '__main__':
    main()
