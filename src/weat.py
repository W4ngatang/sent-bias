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



def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    cossims: mapping from (target_str, attr_str) to cosine similarity
    """
    def cossim(x, y):
        return np.dot(x, y) / math.sqrt(np.dot(x, x)*np.dot(y, y))

    cossims = {}
    for xy in XY:
        for ab in AB:
            cossims[(xy, ab)] = cossim(XY[xy], AB[ab])
    return cossims

def s_wAB(w, A, B, cossims=None):
    """
    "...measures the association of [word] w with the attribute"
    """
    def mean_w_A(w, A, cossims=None):
        """
        Mean cosine similarity of word w across all words in set A.
        """
        total = sum(cossims[(w, a)] for a in A)
        return total / len(A)

    return mean_w_A(w, A, cossims=cossims) - mean_w_A(w, B, cossims=cossims)

def s_XYAB(X, Y, A, B, cossims=None):
    """
    Caliskan: "...measures the differential association of the two sets of
    target words with the attribute."
    Formally, \sum_{x in X} s(x, A, B) - \sum_{y in Y} s(y, A, B)
        where s(x, A, B) = mean_{a in A} cos(x, a) - mean_{b in B} cos(x, b)
    """
    sum_s_xAB = sum(s_wAB(x, A, B, cossims=cossims) for x in X)
    sum_s_yAB = sum(s_wAB(y, A, B, cossims=cossims) for y in Y)
    return sum_s_xAB - sum_s_yAB

def p_val_permutation_test(X, Y, A, B, n_samples, cossims=None):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''

    if cossims: # X, Y, A, B are sets of strings/hashable keys
        assert len(X) == len(Y)
        size = len(X)
        assoc = s_XYAB(X, Y, A, B, cossims=cossims)
        XY = X.union(Y)
        total_true = 1.0
        total = 1.0

        # this way computes all subsets, which is prohibitive for large
        # set sizes
        #for Xi in it.combinations(XY, len(X)):
        #  Yi = XY.difference(Xi)
        #  assert len(Xi) == len(Yi)
        #  if s_XYAB(Xi, Yi, A, B, cossims=cossims) > assoc:
        #    total_true += 1
        #  total += 1

        # instead sample 100K subsets
        if scipy.special.binom(2 * len(X), len(X)) > n_samples:
            XY_list = list(XY)
            while total < n_samples:
                random.shuffle(XY_list)
                Xi = XY_list[:size]
                Yi = XY_list[size:]
                assert len(Xi) == len(Yi)
                if s_XYAB(Xi, Yi, A, B, cossims=cossims) > assoc:
                    total_true += 1
                total += 1
        else:
            for Xi in it.combinations(XY, len(X)):
                Yi = XY.difference(Xi)
                assert len(Xi) == len(Yi)
                if s_XYAB(Xi, Yi, A, B, cossims=cossims) > assoc:
                    total_true += 1
                total += 1

    else: # TODO(Rachel): when is this branch hit?
        assert len(X) == len(Y)
        assoc = s_XYAB(X, Y, A, B)
        XY = X+Y
        total_true = 1.0
        total = 0.0
        for subset in it.combinations(XY, len(X)):
            Xi, Yi = [], []
            for x_or_y in XY:
                if x_or_y in subset:
                    Xi.append(x_or_y)
                else:
                    Yi.append(x_or_y)
            assert len(Xi) == len(Yi) == len(X)
            if s_XYAB(Xi, Yi, A, B) > assoc:
                total_true += 1
            total += 1
    return total_true / total

def effect_size(X, Y, A, B, cossims=None):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) strings
    """
    def mean_s_wAB(X, A, B, cossims):
        return sum(s_wAB(x, A, B, cossims=cossims) for x in X) / len(X)

    def stdev_s_wAB(X, A, B, cossims=None):
        return np.std([s_wAB(x, A, B, cossims=cossims) for x in X]) #ddof=0 or 1?

    XY = X.union(Y)
    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(XY, A, B, cossims=cossims)
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

def run_test(A, B, X, Y, n_samples):
    ''' Run a WEAT.
    args:
        - A, B, X, Y (Dict[str: np.array]): dictionaries mapping words
            to their encodings
    '''
    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    log.info("Computing pval...")
    pval = p_val_permutation_test(set(X), set(Y), set(A), set(B), n_samples, cossims)
    log.info("pval: %.9f", pval)

    log.info("computing effect size...")
    esize = effect_size(set(X), set(Y), set(A), set(B), cossims=cossims)
    log.info("esize: %.9f", esize)
    return esize, pval

if __name__ == "__main__":
    X = {"x"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    Y = {"y"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    A = {"a"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    B = {"b"+str(i):2*np.random.rand(10)-1 for i in range(25)}
    A = X
    B = Y

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    log.info("computing pval...")
    pval = p_val_permutation_test(set(X), set(Y), set(A), set(B), cossims=cossims, n_samples=10000)
    log.info("pval: %.9f", pval)

    log.info("computing effect size...")
    esize = effect_size(set(X), set(Y), set(A), set(B), cossims=cossims)
    log.info("esize: %.9f", esize)

