import numpy as np
from allennlp.commands.elmo import ElmoEmbedder


def encode(sents, time_combine_method="max", layer_combine_method="add"):
    """ Load ELMo and encode sents """
    elmo = ElmoEmbedder()
    vecs = {}
    for sent in sents:
        vec_seq = elmo.embed_sentence(sent)
        if time_combine_method == "max":
            vec = vec_seq.max(axis=1)
        elif time_combine_method == "mean":
            vec = vec_seq.mean(axis=1)
        elif time_combine_method == "concat":
            vec = np.concatenate(vec_seq, axis=1)
        elif time_combine_method == "last":
            vec = vec_seq[:, -1]
        else:
            raise NotImplementedError

        if layer_combine_method == "add":
            vec = vec.sum(axis=0)
        elif layer_combine_method == "mean":
            vec = vec.mean(axis=0)
        elif layer_combine_method == "concat":
            vec = np.concatenate(vec, axis=0)
        elif layer_combine_method == "last":
            vec = vec[-1]
        else:
            raise NotImplementedError
        vecs[' '.join(sent)] = vec
    return vecs
