"""Convert raw GloVe word vector text file to h5."""
import h5py
import numpy as np
import logging


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        'Convert GloVe vectors to HDF5 format for usage with GenSen. '
        'Write HDF5-formatted vectors to same location as GloVe vectors but '
        'with .txt replaced by .h5 .',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_path',
                        help='Path to GloVe vectors (should end with .txt).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    embeddings_index = {}

    if not args.input_path.endswith('.txt'):
        raise Exception('input_path path should have .txt extension')
    output_path = args.input_path[:-len('.txt')] + '.h5'

    logging.info('Reading GloVe vectors from {}'.format(args.input_path))
    f = open(args.input_path)
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs  # .decode()
    f.close()
    vocab, vector = [], []
    for emb in embeddings_index:
        # print(emb)
        vocab.append(emb)
        vector.append(embeddings_index[emb])
    vector = np.asarray(vector)

    logging.info('Writing HDF5-formatted vectors to {}'.format(output_path))
    f = h5py.File(output_path, 'w')
    dt = h5py.special_dtype(vlen=str)     # PY3
    f.create_dataset(data=vector, name='embedding')

    voc = [v.encode('utf-8') for v in vocab]
    f.create_dataset(data=voc, name='words_flatten', dtype=dt)
    f.close()


if __name__ == '__main__':
    main()
