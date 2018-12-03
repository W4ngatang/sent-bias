# sent-bias

## Setup 

First, install Anaconda and a C++ compiler (for example, `g++`) if you
do not have them.  Then
use `environment.yml` to create a conda environment with all necessary
code dependencies: `conda env create -f environment.yml`.
Activate the environment with `source activate sentbias`.

Then, in the environment, download the NLTK punkt tokenization
resource:

```
python -c 'import nltk; nltk.download("punkt")'
```

You will also need to download pretrained model weights for each model
you want to test.  Instructions for each supported model are as
follows.

### Bag-of-words (bow), others

Several models require GloVe words vectors.
Download and unzip the [GloVe Common Crawl 840B 300d
vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) from the
[Stanford NLP GloVe web
page](https://nlp.stanford.edu/projects/glove/).  Make note of the
path to the resultant text file; you will need to pass it to `sentbias/main.py` using the `--glove_path` flag.

### ELMo

ELMo weights will be downloaded and cached at runtime.  Set `ALLENNLP_CACHE_ROOT` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `~/.allennlp`.  For example, if using bash, run this before running ELMo bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

```
export ALLENNLP_CACHE_ROOT=/data/allennlp_cache
```

### Infersent

Download the model checkpoints from the [InferSent repo](https://github.com/facebookresearch/InferSent):

```
wget https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle
wget https://s3.amazonaws.com/senteval/infersent/infersent.snli.pickle
```

Make a note of the directory you download them to; you will need to pass it to `sentbias/main.py` using the `--infersent_dir` flag.

### GenSen

Download the model checkpoints from the [GenSen repo](https://github.com/Maluuba/gensen#setting-up-models--pre-trained-word-vecotrs):

```
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_bothskip_2layer_vocab.pkl
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_bothskip_2layer.model
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_bothskip_parse_vocab.pkl
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_bothskip_parse.model
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_bothskip_vocab.pkl
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_bothskip.model
wget https://genseniclr2018.blob.core.windows.net/models/nli_large_vocab.pkl
wget https://genseniclr2018.blob.core.windows.net/models/nli_large.model
```

Make a note of the directory you download them to; you will need to pass it to `sentbias/main.py` using the `--gensen_dir` flag.

You will also need to process your GloVe word vectors into an HDF5 format.  To do this run `scripts/glove2h5.py` on the path to your GloVe vectors:

```
python scripts/glove2h5.py path/to/glove/vectors.txt
```

### BERT

BERT weights will be downloaded and cached at runtime.  Set `PYTORCH_PRETRAINED_BERT_CACHE` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `~/.pytorch_pretrained_bert`.  For example, if using bash, run this before running BERT bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

```
export PYTORCH_PRETRAINED_BERT_CACHE=/data/bert_cache
```

## Running Bias Tests

We provide a script that demonstrates how to run the bias tests for each model.  To use it, minimally set the path to the GloVe vectors as `GLOVE_PATH` in a file called `user_config.sh`:
 
```
GLOVE_PATH=path/to/glove/vectors.txt
```
 
Then copy `scripts/run_tests.sh` to another location, edit as desired, and run it with `bash`.

### Details

To run bias tests directly, run `main` with one or more tests and one or more models.  Note that each model may require additional command-line flags specifying locations of resources and other options. For example, to run all tests against the bag-of-words (GloVe) and ELMo models:

```
python sentbias/main.py -m bow,elmo
    --glove_path path/to/glove/vectors.txt
```

Run `python sentbias/main.py --help` to see a full list of options.

## Code Tests

To run tests on the code do the following:

```
pytest
flake8
```

## TODO

- track down NaNs in GenSen [Shikha]: looks like the denominator (stddev) of the effect size is 0 because a lot of the vectors are the same...possibly a problem with OOV, but all these words should be in GloVe (base word representations used). Can you make sure the vocab expansion method they implemented is being used?
- add options for concatenation of GenSen models [Shikha]
- implement SkipThought [Shikha]
