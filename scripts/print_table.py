#!/usr/bin/env python3

from csv import DictReader


STAR_THRESHOLD = 0.01
STAR = '^*'
STAR_SPACE = '\phantom{^*}'
SIGNIFICANT_FIGURES = 2

MODELS = (
    ('bow', ''),
    ('infersent', ''),
    ('gensen', 'version=nli_large_bothskip_parse,nli_large_bothskip'),
    ('guse', ''),
    ('elmo', 'time_combine=mean;layer_combine=add'),
    (None, None),
    ('bert', 'version=bert-large-cased'),
)

TESTS = (
    ('weat1', 'C1: Flowers/Insects', 'word'),
    ('sent-weat1', 'C1: Flowers/Insects', 'sent'),
    ('weat2', 'C2: Instruments/Weapons', 'word'),
    ('sent-weat2', 'C2: Instruments/Weapons', 'sent'),
    ('weat3', 'C3: EA/AA Names', 'word'),
    ('sent-weat3', 'C3: EA/AA Names', 'sent'),
    ('weat6', 'C6: M/F Names, Career', 'word'),
    ('sent-weat6', 'C6: M/F Names, Career', 'sent'),
    (None, None, None),
    ('angry_black_woman_stereotype', 'ABWS', 'word'),
    ('sent-angry_black_woman_stereotype', 'ABWS', 'sent'),
    (None, None, None),
    ('heilman_double_bind_competent_one_word', 'DB: Competent', 'word'),
    ('heilman_double_bind_competent_one_sentence', 'DB: Competent', 'sent'),
    ('heilman_double_bind_competent_1-', 'DB: Competent', 'sent+'),
    ('heilman_double_bind_likable_one_word', 'DB: Likable', 'word'),
    ('heilman_double_bind_likable_one_sentence', 'DB: Likable', 'sent'),
    ('heilman_double_bind_likable_1-', 'DB: Likable', 'sent+'),
)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'Read results from tsv file and print out rows for LaTeX table.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('results_path',
                        help='Paths to results tsv file.')
    args = parser.parse_args()

    results = dict()
    with open(args.results_path) as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['model'], row['options'], row['test'])
            if key in results:
                raise Exception('duplicate key: {}'.format(key))
            results[key] = row

    for (test_name, test_description, context_level) in TESTS:
        if test_name is None:
            print('\n\\midrule\n')
        else:
            print('{} & {}'.format(test_description, context_level), end='')
            for (model, options) in MODELS:
                if model is None:
                    print(' & ', end='')
                else:
                    row = results[(model, options, test_name)]
                    effect_size = float(row['effect_size'])
                    star = STAR if float(row['p_value']) <= STAR_THRESHOLD else STAR_SPACE
                    print(' & ${effect_size:.{precision}f}{star}$'.format(
                        effect_size=effect_size,
                        star=star,
                        precision=SIGNIFICANT_FIGURES), end='')
            print(' \\\\')


if __name__ == '__main__':
    main()
