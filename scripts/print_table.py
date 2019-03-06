#!/usr/bin/env python3

from csv import DictReader
import logging


STAR_THRESHOLD = 0.01
SINGLE_STAR = r'^{*\phantom{*}}'
DOUBLE_STAR = r'^{**}'
STAR_SPACE = r'\phantom{^{**}}'

SIGNIFICANT_FIGURES = 2

CTX_WORD = 'word'
CTX_SENT = 'sent'
CTX_SENT_UNBLEACHED = 'sent (u)'
CTX_PARA = 'para'

MODEL_SETS = {
    'overall': (
        ('bow', ''),
        ('infersent', ''),
        ('gensen', 'version=nli_large_bothskip_parse,nli_large_bothskip'),
        ('guse', ''),
        ('elmo', 'time_combine=mean;layer_combine=add'),
        ('openai', ''),
        ('bert', 'version=bert-large-cased'),
    ),
    '78': (
        ('bow', ''),
        ('gensen', 'version=nli_large_bothskip_parse,nli_large_bothskip'),
        ('openai', ''),
        ('bert', 'version=bert-large-cased'),
    ),
    '345': (
        ('bow', ''),
        ('elmo', 'time_combine=mean;layer_combine=add'),
    ),
}

DEFAULT_MODEL_SET = 'overall'

TEST_SETS = {
    'overall': (
        ('weat1', 'C1: Flowers/Insects', CTX_WORD),
        ('sent-weat1', 'C1: Flowers/Insects', CTX_SENT),
        ('weat3', 'C3: EA/AA Names', CTX_WORD),
        ('sent-weat3', 'C3: EA/AA Names', CTX_SENT),
        ('weat6', 'C6: M/F Names, Career', CTX_WORD),
        ('sent-weat6', 'C6: M/F Names, Career', CTX_SENT),
        ('angry_black_woman_stereotype', 'ABW Stereotype', CTX_WORD),
        ('sent-angry_black_woman_stereotype', 'ABW Stereotype', CTX_SENT),
        ('heilman_double_bind_competent_one_word', 'Double Bind: Competent', CTX_WORD),
        ('sent-heilman_double_bind_competent_one_word', 'Double Bind: Competent', CTX_SENT),
        ('heilman_double_bind_competent_one_sentence', 'Double Bind: Competent', CTX_SENT_UNBLEACHED),
        ('heilman_double_bind_likable_one_word', 'Double Bind: Likable', CTX_WORD),
        ('sent-heilman_double_bind_likable_one_word', 'Double Bind: Likable', CTX_SENT),
        ('heilman_double_bind_likable_one_sentence', 'Double Bind: Likable', CTX_SENT_UNBLEACHED),
    ),
    '78': (
        ('weat7', 'C7: Math/Arts, M/F', CTX_WORD),
        ('sent-weat7', 'C7: Math/Arts, M/F', CTX_SENT),
        ('weat8', 'C8: Science/Arts, M/F', CTX_WORD),
        ('sent-weat8', 'C8: Science/Arts, M/F', CTX_SENT),
    ),
    '345': (
        ('weat3', 'C3: EA/AA Names (32/25)', CTX_WORD),
        ('sent-weat3', 'C3: EA/AA Names (32/25 +)', CTX_SENT),
        ('weat4', 'C4: EA/AA Names (16/25)', CTX_WORD),
        ('sent-weat4', 'C4: EA/AA Names (16/25 +)', CTX_SENT),
        ('weat5', 'C5: EA/AA Names (16/8)', CTX_WORD),
        ('sent-weat5', 'C5: EA/AA Names (16/8 +)', CTX_SENT),
    ),
    'caliskan': (
        ('weat1', 'C1: Flowers/Insects', CTX_WORD),
        ('sent-weat1', 'C1: Flowers/Insects', CTX_SENT),
        ('weat2', 'C2: Instruments/Weapons', CTX_WORD),
        ('sent-weat2', 'C2: Instruments/Weapons', CTX_SENT),
        ('weat3', 'C3: EA/AA Names', CTX_WORD),
        ('sent-weat3', 'C3: EA/AA Names', CTX_SENT),
        ('weat4', 'C4: EA/AA Names', CTX_WORD),
        ('sent-weat4', 'C4: EA/AA Names', CTX_SENT),
        ('weat5', 'C5: EA/AA Names', CTX_WORD),
        ('sent-weat5', 'C5: EA/AA Names', CTX_SENT),
        ('weat6', 'C6: M/F Names, Career', CTX_WORD),
        ('sent-weat6', 'C6: M/F Names, Career', CTX_SENT),
        ('weat7', 'C7: Math/Arts, M/F', CTX_WORD),
        ('sent-weat7', 'C7: Math/Arts, M/F', CTX_SENT),
        ('weat8', 'C8: Science/Arts, M/F', CTX_WORD),
        ('sent-weat8', 'C8: Science/Arts, M/F', CTX_SENT),
        ('weat9', 'C9: Ment/Phys, Temp/Perm', CTX_WORD),
        ('sent-weat9', 'C9: Ment/Phys, Temp/Perm', CTX_SENT),
        ('weat10', 'C10: Young/Old Names', CTX_WORD),
        ('sent-weat10', 'C10: Young/Old Names', CTX_SENT),
    ),
}

DEFAULT_TEST_SET = 'overall'


def holm_bonferroni(results):
    results_list = sorted(results.items(), key=lambda p: p[1]['p_value'])
    reject = True
    for (i, (k, _)) in enumerate(results_list):
        row = results[k]
        if row['p_value'] > STAR_THRESHOLD / (len(results_list) - i):
            reject = False
        row['reject'] = reject
    return results


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'Read results from tsv file and print out rows for LaTeX table.',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('results_path',
                        help='Paths to results tsv file.')
    parser.add_argument('--correct_within_groups', action='store_true',
                        help='Correct for multiple tests within each test group (Caliskan, ABW, DB)')
    parser.add_argument('--test_set', choices=TEST_SETS.keys(), default=DEFAULT_TEST_SET,
                        help='Name of set of tests to report.')
    parser.add_argument('--model_set', choices=MODEL_SETS.keys(), default=DEFAULT_MODEL_SET,
                        help='Name of set of models to report.')
    parser.add_argument('--p_values_only', action='store_true',
                        help='Report p-values only (default: effect sizes with stars).')
    parser.add_argument('--header', action='store_true',
                        help='Print header row.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = dict()
    with open(args.results_path) as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            for k in ('p_value', 'effect_size'):
                row[k] = float(row[k])
            for k in ('num_targ1', 'num_targ2', 'num_attr1', 'num_attr2'):
                row[k] = int(row[k])

            k = (row['model'], row['options'], row['test'])
            if k in results:
                raise Exception('duplicate key: {}'.format(k))
            results[k] = row

    if args.correct_within_groups:
        logging.info('Computing multiple testing correction within test groups')
        caliskan_results = holm_bonferroni(dict(
            ((model, options, test), v) for ((model, options, test), v) in results.items()
            if 'weat' in test))
        abw_results = holm_bonferroni(dict(
            ((model, options, test), v) for ((model, options, test), v) in results.items()
            if 'angry_black_woman' in test))
        db_results = holm_bonferroni(dict(
            ((model, options, test), v) for ((model, options, test), v) in results.items()
            if 'double_bind' in test))

        if len(results) != len(caliskan_results) + len(abw_results) + len(db_results):
            raise Exception('number of results does not match sum across groups (after correction)')

        results = caliskan_results
        results.update(abw_results)
        results.update(db_results)
    else:
        results = holm_bonferroni(results)

    if args.header:
        print('Test & Context', end='')
        for (model, options) in MODEL_SETS[args.model_set]:
            if model is None:
                print(' & ', end='')
            else:
                print(' & {}'.format(model), end='')
        print(' \\\\')

    for (test_name, test_description, context_level) in TEST_SETS[args.test_set]:
        if test_name is None:
            print('\n\\midrule\n')
        else:
            print('{} & {}'.format(test_description, context_level), end='')
            for (model, options) in MODEL_SETS[args.model_set]:
                if model is None:
                    print(' & ', end='')
                else:
                    row = results[(model, options, test_name)]
                    p_value = row['p_value']
                    effect_size = row['effect_size']
                    star = DOUBLE_STAR if row['reject'] else (
                        SINGLE_STAR if p_value <= STAR_THRESHOLD else STAR_SPACE
                    )
                    if args.p_values_only:
                        p_value_str = '{p_value:.{precision}g}'.format(
                            p_value=p_value,
                            precision=SIGNIFICANT_FIGURES)
                        if 'e' in p_value_str:
                            (base, exponent) = p_value_str.split('e')
                            (sign, exponent) = (exponent[0], exponent[1:])
                            exponent = exponent.lstrip('0')
                            p_value_str = r'{} \times 10^{{{}{}}}'.format(base, sign, exponent)
                        print(' & ${}$'.format(p_value_str), end='')
                    else:
                        print(' & ${effect_size:.{precision}f}{star}$'.format(
                            effect_size=effect_size,
                            star=star,
                            precision=SIGNIFICANT_FIGURES), end='')
            print(' \\\\')


if __name__ == '__main__':
    main()
