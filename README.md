# sent-bias

This repository contains the code and data for the paper "[On Measuring Social Biases in Sentence Encoders](https://arxiv.org/abs/1903.10561)" by Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman and Rachel Rudinger.

## Setup

### Environment setup

First, install Anaconda and a C++ compiler (for example, `g++`) if you
do not have them.

#### Using the prespecified environment

Use `environment.yml` to create a conda environment with all necessary
code dependencies:

```
conda env create -f environment.yml
```

Activate the environment as follows:

```
source activate sentbias
```

#### Recreating the environment

Alternatively (for example,
if you have problems using the prespecified environment), follow
approximately the following steps to recreate it.  First, create a new
environment with Python 3.6:

```
conda create -n sentbias python=3.6
```

Then activate the environment and add the remaining dependencies:

```
source activate sentbias
conda install pytorch=0.4.1 cuda90 -c pytorch
conda install tensorflow
pip install allennlp gensim tensorflow-hub pytorch-pretrained-bert numpy scipy nltk spacy h5py scikit-learn
```

#### Environment postsetup

Now, with the environment activated, download the NLTK punkt and spacy en
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

BERT weights will be downloaded from [Bert repo](https://github.com/huggingface/pytorch-pretrained-BERT) and cached at runtime.  Set `PYTORCH_PRETRAINED_BERT_CACHE` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `~/.pytorch_pretrained_bert`.  For example, if using bash, run this before running BERT bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

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

To produce the GPT representations of the SEAT data at last, run `extract_repr.py` on those tasks (this may take a while):

```
python extract_repr.py --config config/bias.conf --overrides "target_tasks = \"$target_tasks\", exp_name = sentbias-openai, run_name = openai, word_embs = none, elmo = 0, openai_transformer = 1, sent_enc = \"null\", skip_embs = 1, sep_embs_for_skip = 1, allow_missing_task_map = 1, combine_method = last"
```

The representations will be saved to files of the form `TASK.encs` (for example, `angry_black_woman_stereotype-openai.encs`) in the directory `$JIANT_PROJECT_PREFIX/sentbias-openai/openai`.  To apply SEAT To them, we first need to strip the `-openai` part from the filenames:

```
for f in $JIANT_PROJECT_PREFIX/sentbias-openai/openai/*-openai.encs
do
    mv $f ${f%-openai.encs}.encs
done
```

Finally, pass the directory path `$JIANT_PROJECT_PREFIX/sentbias-openai/openai` to `sentbias/main.py` using the `--openai_encs` flag.

### ELMo

ELMo weights will be downloaded from [allennlp repo](https://github.com/allenai/allennlp/tree/master/allennlp)  and cached at runtime.  Set `ALLENNLP_CACHE_ROOT` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `~/.allennlp`.  For example, if using bash, run this before running ELMo bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

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

Download the AllNLI [InferSent](https://github.com/facebookresearch/InferSent) model checkpoint (Facebook has deleted this version; we are temporarily hosting a copy):

```
wget http://sent-bias.s3-website-us-east-1.amazonaws.com/infersent.allnli.pickle
```

Make a note of the directory you download them to; you will need to pass it to `sentbias/main.py` using the `--infersent_dir` flag.

### Universal Sentence Encoder (Google)

Universal Sentence Encoder weights will be downloaded from [Universal Sentence Encoder repo](https://github.com/tensorflow/tfjs-models/tree/master/universal-sentence-encoder) and cached at runtime.  Set `TFHUB_CACHE_DIR` in your environment to a directory you'd like them to be saved to; otherwise they will be saved to `/tmp/tfhub_modules`.  For example, if using bash, run this before running Universal Sentence Encoder bias tests or put it in your `~/.bashrc` and start a new shell session to run bias tests:

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
python sentbias/main.py -m bow,elmo --glove_path path/to/glove.840B.300d.txt
```

If they are available, cached sentence representations in the `output` directory will be loaded and used; if they are not available, they will be computed (and cached under `output`).
Run `python sentbias/main.py --help` to see a full list of options.

## Code Tests

To run style checks, first install `flake8`:

```
pip install flake8
```

Then run it as follows:

```
flake8
```

## License

This code is distributed under the Creative Commons
Attribution-NonCommercial 4.0 International license, which can be found
in the `LICENSE` file in this directory.

The file `sentbias/models.py` is based on [`models.py` in InferSent](https://github.com/facebookresearch/InferSent/blob/74990f5f9aa46d2e549eeb7b80bd64dbf338407d/models.py) with small modifications by us (May, Wang, Bordia, Bowman, and Rudinger); the original file is copyright Facebook, Inc. under the Creative Commons Attribution-NonCommercial 4.0 International license.


The file `sentbias/encoders/gensen.py` is based on [`gensen.py` in gensen](https://github.com/Maluuba/gensen/blob/8e6948af62c4b9b1ba77bc000abf70ab68b4663d/gensen.py) with small modifications by us (May, Wang, Bordia, Bowman, and Rudinger); the original file is copyright Microsoft Corporation under the MIT license.
