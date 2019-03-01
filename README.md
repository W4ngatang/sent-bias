# sent-bias

## Setup

First, install Anaconda and a C++ compiler (for example, `g++`) if you
do not have them.  Then
use `environment.yml` to create a conda environment with all necessary
code dependencies:

```
conda env create -f environment.yml
```

Activate the environment:

```
source activate sentbias
```

Now, in the environment, download the NLTK punkt and spacy en
resources:

```
python -c 'import nltk; nltk.download("punkt")'
python -m spacy download en
```

You will also need to download pretrained model weights for each model
you want to test.  Instructions for each supported model are as
follows.

### Bag-of-words (bow); also GenSen and InferSent

Several models require GloVe words vectors.  Download and unzip the
GloVe Common Crawl 840B 300d vectors from the [Stanford NLP GloVe web
page](https://nlp.stanford.edu/projects/glove/):

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

Make note of the path to the resultant text file; you will need to pass
it to `sentbias/main.py` using the `--glove_path` flag.

### BERT

BERT weights will be downloaded and cached at runtime.  Set `PYTORCH_PRETRAINED_BERT_CACHE` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `~/.pytorch_pretrained_bert`.  For example, if using bash, run this before running BERT bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

```
export PYTORCH_PRETRAINED_BERT_CACHE=/data/bert_cache
```

### GPT (OpenAI)

Evaluation of GPT is supported through the [jiant](https://github.com/jsalt18-sentence-repl/jiant) project.  To generate GPT predictions for evaluating in SEAT, first initialize and update the jiant code:

```
git submodule update --init --recursive
```

With jiant initialized, change your current directory to `jiant`.  The rest of the commands in this section should be run in that directory.

```
cd jiant
```

Now create a conda environment with core jiant dependencies:

```
conda env create -f environment.yml
```

Activate that environment and collect the remaining Python dependencies:

```
source activate jiant
python -m nltk.downloader perluniprops nonbreaking_prefixes punkt
pip install python-Levenshtein ftfy
conda install tensorflow
python -m spacy download en
```

Next we need to set a few environment variables.  Change the value of `ROOT_DIR` to a directory on a filesystem with at least six gigabytes of free space; the directory will be created if it doesn't exist:

```
ROOT_DIR="/media/cjmay/Data1/sent-bias"

export JIANT_DATA_DIR=$ROOT_DIR/jiant
export NFS_PROJECT_PREFIX="$ROOT_DIR/ckpts/jiant"
export JIANT_PROJECT_PREFIX="$ROOT_DIR/ckpts/jiant"
WORD_EMBS_DIR="$ROOT_DIR/fasttext"
export WORD_EMBS_FILE="$WORD_EMBS_DIR/crawl-300d-2M.vec"
export FASTTEXT_MODEL_FILE=None
```

Download [fasttext vectors](https://fasttext.cc/docs/en/english-vectors.html):

```
mkdir -p $WORD_EMBS_DIR
curl -L https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip -o ${WORD_EMBS_FILE}.zip
unzip -d $WORD_EMBS_DIR ${WORD_EMBS_FILE}.zip
```

Retokenize the SEAT tests using BPE:

```
mkdir -p $JIANT_DATA_DIR
cp -r ../tests $JIANT_DATA_DIR/WEAT
python probing/retokenize_weat_data.openai.py $JIANT_DATA_DIR/WEAT/*.jsonl
```

Now we put a comma-separated list of the jiant *tasks* we want to run in `target_tasks`:

```
target_tasks=angry_black_woman_stereotype-openai,angry_black_woman_stereotype_b-openai,sent-angry_black_woman_stereotype-openai,sent-angry_black_woman_stereotype_b-openai,heilman_double_bind_competent_1-openai,heilman_double_bind_competent_1+3--openai,heilman_double_bind_competent_1--openai,heilman_double_bind_competent_one_sentence-openai,heilman_double_bind_competent_one_word-openai,sent-heilman_double_bind_competent_one_word-openai,heilman_double_bind_likable_1-openai,heilman_double_bind_likable_1+3--openai,heilman_double_bind_likable_1--openai,heilman_double_bind_likable_one_sentence-openai,heilman_double_bind_likable_one_word-openai,sent-heilman_double_bind_likable_one_word-openai,weat1-openai,weat2-openai,weat3-openai,weat3b-openai,weat4-openai,weat5-openai,weat5b-openai,weat6b-openai,weat6-openai,weat7-openai,weat7b-openai,weat8-openai,weat8b-openai,weat9-openai,weat10-openai,sent-weat1-openai,sent-weat2-openai,sent-weat3-openai,sent-weat3b-openai,sent-weat4-openai,sent-weat5-openai,sent-weat5b-openai,sent-weat6-openai,sent-weat6b-openai,sent-weat7-openai,sent-weat7b-openai,sent-weat8-openai,sent-weat8b-openai,sent-weat9-openai,sent-weat10-openai
```

Finally, to produce the GPT representations of the SEAT data, run `extract_repr.py` on those tasks (this may take a while):

```
python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"$target_tasks\", exp_name = sentbias-openai, run_name = openai, word_embs = none, elmo = 0, openai_transformer = 1, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = last"
```

The path to the directory containing the representations should be passed to `sentbias/main.py` using the `--openai_encs` flag.

### ELMo

ELMo weights will be downloaded and cached at runtime.  Set `ALLENNLP_CACHE_ROOT` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `~/.allennlp`.  For example, if using bash, run this before running ELMo bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

```
export ALLENNLP_CACHE_ROOT=/data/allennlp_cache
```

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
python scripts/glove2h5.py path/to/glove.840B.300d.txt
```

### Infersent

Download the model checkpoints from the [InferSent repo](https://github.com/facebookresearch/InferSent):

```
wget https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle
wget https://s3.amazonaws.com/senteval/infersent/infersent.snli.pickle
```

Make a note of the directory you download them to; you will need to pass it to `sentbias/main.py` using the `--infersent_dir` flag.

### Universal Sentence Encoder (Google)

Universal Sentence Encoder weights will be downloaded and cached at runtime.  Set `TFHUB_CACHE_DIR` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `/tmp/tfhub_modules`.  For example, if using bash, run this before running Universal Sentence Encoder bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

```
export TFHUB_CACHE_DIR=/data/tfhub_cache
```

## Running Bias Tests

We provide a script that demonstrates how to run the bias tests for each model.  To use it, minimally set the path to the GloVe vectors as `GLOVE_PATH` in a file called `user_config.sh`:

```
GLOVE_PATH=path/to/glove.840B.300d.txt
```

Then copy `scripts/run_tests.sh` to a temporary location, edit as desired, and run it with `bash`.

### Details

To run bias tests directly, run `main` with one or more tests and one or more models.  Note that each model may require additional command-line flags specifying locations of resources and other options. For example, to run all tests against the bag-of-words (GloVe) and ELMo models:

```
python sentbias/main.py -m bow,elmo --glove_path $GLOVE_PATH
```

By default, output will be written to `output` in the current directory.
Run `python sentbias/main.py --help` to see a full list of options.

## Code Tests

To run tests on the code do the following:

```
pytest
flake8
```
