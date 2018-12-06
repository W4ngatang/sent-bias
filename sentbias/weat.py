''' Implements the WEAT tests '''
import logging as log
import math
import random
import itertools as it
import numpy as np
import scipy.special

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.


def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """

    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims


def s_XYAB(X, Y, A, B, cossims):
    r"""
    Caliskan: "...measures the differential association of the two sets of
    target words with the attribute."
    Formally, \sum_{x in X} s(x, A, B) - \sum_{y in Y} s(y, A, B)
        where s(x, A, B) = mean_{a in A} cos(x, a) - mean_{b in B} cos(x, b)
    """
    s_wAB = cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)
    return s_wAB[X].sum() - s_wAB[Y].sum()


def p_val_permutation_test(X, Y, A, B, n_samples, cossims):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    assert len(X) == len(Y)
    size = len(X)
    assoc = s_XYAB(X, Y, A, B, cossims=cossims)
    XY = X + Y
    total_true = 0
    total_equal = 0
    total = 0

    if scipy.special.binom(2 * len(X), len(X)) > n_samples:
        # We only have as much precision as the number of samples drawn;
        # bias the p-value (hallucinate a positive observation) to
        # reflect that.
        total_true += 1
        total += 1
        log.info('Drawing {} samples (and biasing by 1)'.format(n_samples - total))
        while total < n_samples:
            random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            s = s_XYAB(Xi, Yi, A, B, cossims=cossims)
            if s > assoc:
                total_true += 1
            elif s == assoc:  # use conservative test
                total_true += 1
                total_equal += 1
            total += 1
    else:
        log.info('Using exact test')
        XY_set = set(XY)
        for Xi in it.combinations(XY, len(X)):
            Xi = list(Xi)
            Yi = list(XY_set.difference(Xi))
            assert len(Xi) == len(Yi)
            s = s_XYAB(Xi, Yi, A, B, cossims=cossims)
            if s > assoc:
                total_true += 1
            elif s == assoc:  # use conservative test
                total_true += 1
                total_equal += 1
            total += 1

    if total_equal:
        log.warning('Equalities contributed {}/{} to p-value'.format(total_equal, total))

    return total_true / total


def mean_s_wAB(X, A, B, cossims):
    return np.mean(cossims[X][:, A].mean(axis=1) - cossims[X][:, B].mean(axis=1))


def stdev_s_wAB(X, A, B, cossims):
    return np.std(cossims[X][:, A].mean(axis=1) - cossims[X][:, B].mean(axis=1), ddof=1)


def effect_size(X, Y, A, B, cossims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X = list(X)
    Y = list(Y)
    A = list(A)
    B = list(B)

    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )


# def run_test(A, B, X, Y, names, n_samples):
def run_test(encs, n_samples):
    ''' Run a WEAT.
    args:
        - encs (Dict[str: Dict]): dictionary mapping targ1, targ2, attr1, attr2
            to dictionaries containing the category and the encodings
        - n_samples (int): number of samples to draw to estimate p-value
            (use exact test if number of permutations is less than or
            equal to n_samples)
    '''
    X, Y = encs["targ1"]["encs"], encs["targ2"]["encs"]
    A, B = encs["attr1"]["encs"], encs["attr2"]["encs"]

    # First convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    log.info("Computing cosine similarities...")
    cossims = construct_cossim_lookup(XY, AB)

    log.info("Null hypothesis: no difference between %s and %s in association to attributes %s and %s",
             encs["targ1"]["category"], encs["targ2"]["category"],
             encs["attr1"]["category"], encs["attr2"]["category"])
    log.info("Computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, n_samples, cossims=cossims)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    log.info("esize: %g", esize)
    return esize, pval


if __name__ == "__main__":
    X = {"x" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    Y = {"y" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = {"a" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    B = {"b" + str(i): 2 * np.random.rand(10) - 1 for i in range(25)}
    A = X
    B = Y

    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    log.info("computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, cossims=cossims, n_samples=10000)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    log.info("esize: %g", esize)
