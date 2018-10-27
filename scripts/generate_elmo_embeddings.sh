source activate sentbias

# ./generate_elmo_embeddings.sh weat1

# Generate elmo embeddings for each concept group (A, B, X, Y), one file per.

allennlp elmo ../elmo/$1.A.txt ../elmo/$1.A.elmo.hdf5 --all --use-sentence-keys && rm ../elmo/std*.log
allennlp elmo ../elmo/$1.B.txt ../elmo/$1.B.elmo.hdf5 --all --use-sentence-keys && rm ../elmo/std*.log
allennlp elmo ../elmo/$1.X.txt ../elmo/$1.X.elmo.hdf5 --all --use-sentence-keys && rm ../elmo/std*.log
allennlp elmo ../elmo/$1.Y.txt ../elmo/$1.Y.elmo.hdf5 --all --use-sentence-keys && rm ../elmo/std*.log
