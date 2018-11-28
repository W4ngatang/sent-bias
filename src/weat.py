''' Implements the WEAT tests '''
import logging as log
import math
import random
import itertools as it
import numpy as np
import scipy.special

import encoders.glove as glove
import encoders.elmo as elmo

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.


def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x)*np.dot(y, y))


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
    """
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
    total = 0

    if scipy.special.binom(2 * len(X), len(X)) > n_samples:
        # We only have as much precision as the number of samples drawn;
        # bias the p-value (hallucinate a positive observation) to
        # reflect that.
        total_true += 1
        total += 1
        log.info('Drawing {} samples'.format(n_samples))
        while total < n_samples:
            random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            if s_XYAB(Xi, Yi, A, B, cossims=cossims) > assoc:
                total_true += 1
            total += 1
    else:
        log.info('Using exact test')
        XY_set = set(XY)
        for Xi in it.combinations(XY, len(X)):
            Yi = list(XY_set.difference(Xi))
            assert len(Xi) == len(Yi)
            if s_XYAB(Xi, Yi, A, B, cossims=cossims) > assoc:
                total_true += 1
            total += 1

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


def load_weat_test(weatname, path=None):
    ''' Load pre-extracted GloVe vectors '''
    if path is None:
        path = "../tests/"
    else:
        path = path.rstrip('/')+'/'

    A = glove.load_glove_file(path+weatname+".A.vec")
    B = glove.load_glove_file(path+weatname+".B.vec")
    X = glove.load_glove_file(path+weatname+".X.vec")
    Y = glove.load_glove_file(path+weatname+".Y.vec")
    return (A, B, X, Y)


def load_elmo_weat_test(weatname, path=None):
    ''' Load pre-computed ELMo vectors '''
    if path is None:
        path = "../encodings/elmo/"
    else:
        path = path.rstrip('/')+'/'
    A = elmo.load_elmo_hdf5(path+weatname+".A.elmo.hdf5")
    B = elmo.load_elmo_hdf5(path+weatname+".B.elmo.hdf5")
    X = elmo.load_elmo_hdf5(path+weatname+".X.elmo.hdf5")
    Y = elmo.load_elmo_hdf5(path+weatname+".Y.elmo.hdf5")
    return (A, B, X, Y)


def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )


def run_test(A, B, X, Y, n_samples):
    ''' Run a WEAT.
    args:
        - A, B, X, Y (Dict[str: np.array]): dictionaries mapping words
            to their encodings
    '''
    # First convert all keys to ints to facilitate array lookups
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    log.info("Computing cosine similarities...")
    cossims = construct_cossim_lookup(XY, AB)

    log.info("Computing pval...")
    pval = p_val_permutation_test(X, Y, A, B, n_samples, cossims=cossims)
    log.info("pval: %g", pval)

    log.info("computing effect size...")
    esize = effect_size(X, Y, A, B, cossims=cossims)
    log.info("esize: %g", esize)
    return esize, pval

if __name__ == "__main__":
    X = {"x"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    Y = {"y"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    A = {"a"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    B = {"b"+str(i):2*np.random.rand(10)-1 for i in range(25)}
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

