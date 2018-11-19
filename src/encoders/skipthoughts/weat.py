''' Implements the WEAT tests '''
import logging as log
import math
import random
import itertools as it
import numpy as np

import glove
#import elmo

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.

#X = [2*np.random.rand(10)-1 for _ in range(5)]
#Y = [2*np.random.rand(10)-1 for _ in range(5)]
#A = [2*np.random.rand(10)-1 for _ in range(4)]
#B = [2*np.random.rand(10)-1 for _ in range(7)]

def cossim(x, y):
    assert type(x)==type(y)
    #print(x.shape, y.shape)
    a =x.dot(y)
    b = x.dot(x) * y.dot(y)
    return  a/math.sqrt(b)

def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    targ_att: mapping from (target_str, attr_str) to cosine similarity
    """
    COSSIMS = {}
    for xy in XY:
        for ab in AB:
            COSSIMS[(xy, ab)] = cossim(XY[xy], AB[ab])
    return COSSIMS

def mean_w_A(w, A, COSSIMS=None):
    """
    Mean cosine similarity of word w across all words
    in set A.
    """
    if COSSIMS:
        total = sum(COSSIMS[(w, a)] for a in A)
        return total / len(A)
    else:
        total = sum(cossim(w, a) for a in A)
        return total / len(A)

def s_wAB(w, A, B, COSSIMS=None):
    """
    "...measures the association of [word] w with the attribute"
    """
    if COSSIMS:
        return mean_w_A(w, A, COSSIMS=COSSIMS) - mean_w_A(w, B, COSSIMS=COSSIMS)
    else:
        return mean_w_A(w, A) - mean_w_A(w, B)

def mean_s_wAB(X, A, B, COSSIMS=None):
    if COSSIMS:
        return sum(s_wAB(x, A, B, COSSIMS=COSSIMS) for x in X) / len(X)
    else:
        raise NotImplementedError

def stdev_s_wAB(X, A, B, COSSIMS=None):
    if COSSIMS:
        return np.std([s_wAB(x, A, B, COSSIMS=COSSIMS) for x in X]) #ddof=0 or 1?
    else:
        raise NotImplementedError

def effect_size(X, Y, A, B, COSSIMS=None):
    """
    X, Y, A, B : sets of target (X, Y) and attribute (A, B) strings
    """
    if COSSIMS:
        XY = X.union(Y)
        numerator = mean_s_wAB(X, A, B, COSSIMS=COSSIMS) - mean_s_wAB(Y, A, B, COSSIMS=COSSIMS)
        denominator = stdev_s_wAB(XY, A, B, COSSIMS=COSSIMS)
        return numerator/denominator
    else:
        raise NotImplementedError

def s_XYAB(X, Y, A, B, COSSIMS=None):
    """
    "...measures the differential association of the two sets of
    target words with the attribute."
    """
    if COSSIMS:
        sum_s_xAB = sum(s_wAB(x, A, B, COSSIMS=COSSIMS) for x in X)
        sum_s_yAB = sum(s_wAB(y, A, B, COSSIMS=COSSIMS) for y in Y)
        return sum_s_xAB - sum_s_yAB
    else:
        sum_s_xAB = sum(s_wAB(x, A, B) for x in X)
        sum_s_yAB = sum(s_wAB(y, A, B) for y in Y)
        return sum_s_xAB - sum_s_yAB

def p_val_permutation_test(X, Y, A, B, COSSIMS=None):

    if COSSIMS:
        # X, Y, A, B are sets of strings/hashable keys
        assert len(X) == len(Y)
        assoc = s_XYAB(X, Y, A, B, COSSIMS=COSSIMS)
        XY = X.union(Y)
        total_true = 1.0
        total = 1.0

        # this way computes all subsets, which is prohibitive for large
        # set sizes
        #for Xi in it.combinations(XY, len(X)):
        #  Yi = XY.difference(Xi)
        #  assert len(Xi) == len(Yi)
        #  if s_XYAB(Xi, Yi, A, B, COSSIMS=COSSIMS) > assoc:
        #    total_true += 1
        #  total += 1

        # instead sample 100K subsets
        XY_list = list(XY)
        while total < 100000:
            random.shuffle(XY_list)
            Xi = XY_list[:len(X)]
            Yi = XY_list[len(X):]
            assert len(Xi) == len(Yi)
            if s_XYAB(Xi, Yi, A, B, COSSIMS=COSSIMS) > assoc:
                total_true += 1
            total += 1
        return total_true / total

    else:
        assert len(X) == len(Y)
        assoc = s_XYAB(X, Y, A, B)
        XY = X+Y
        total_true = 1.0
        total = 0.0
        for subset in it.combinations(XY, len(X)):
            Xi, Yi = [], []
            for x_or_y in XY:
                if x_or_y in subset:
                    #---> 49       if x_or_y in subset:
                    #     50         Xi.append(x_or_y)
                    #     51       else:
                    #
                    #ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                    Xi.append(x_or_y)
                else:
                    Yi.append(x_or_y)
            assert len(Xi) == len(Yi) == len(X)
            if s_XYAB(Xi, Yi, A, B) > assoc:
                total_true += 1
            total += 1
        return total_true / total

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
        path = "../elmo/"
    else:
        path = path.rstrip('/')+'/'
    A = elmo.load_elmo_hdf5(path+weatname+".A.elmo.hdf5")
    B = elmo.load_elmo_hdf5(path+weatname+".B.elmo.hdf5")
    X = elmo.load_elmo_hdf5(path+weatname+".X.elmo.hdf5")
    Y = elmo.load_elmo_hdf5(path+weatname+".Y.elmo.hdf5")
    return (A, B, X, Y)

def run_test(A, B, X, Y):
    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)
    
    COSSIMS = construct_cossim_lookup(XY, AB)
    log.info("computing pval...")
    pval = p_val_permutation_test(set(X), set(Y), set(A), set(B), COSSIMS=COSSIMS)
    log.info("pval: %.9f", pval)

    log.info("computing effect size...")
    
    esize = effect_size(set(X), set(Y), set(A), set(B), COSSIMS=COSSIMS)
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

    COSSIMS = construct_cossim_lookup(XY, AB)
    log.info("computing pval...")
    pval = p_val_permutation_test(set(X), set(Y), set(A), set(B), COSSIMS=COSSIMS)
    log.info("pval: %.9f", pval)

    log.info("computing effect size...")
    esize = effect_size(set(X), set(Y), set(A), set(B), COSSIMS=COSSIMS)
    log.info("esize: %.9f", esize)


