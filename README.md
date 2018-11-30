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

### GloVe 

Download and unzip the [GloVe Common Crawl 840B 300d
vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) from the
[Stanford NLP GloVe web
page](https://nlp.stanford.edu/projects/glove/).  Make note of the
path to the resultant text file.

### ELMo

```
cd scripts
./weat_glove2elmo.py $WEATTESTNAME
./generate_elmo_embeddings.sh $WEATTESTNAME
# for example
./weat_glove2elmo.py weat1
./generate_elmo_embeddings.sh weat1
```

### Infersent

Download the model checkpoints from the [original repo](https://github.com/facebookresearch/InferSent) and put them in `src/encoders`.

### GenSen

Download the model checkpoints.
Note, you need to process your GloVe word vectors into an HDF5 format. Run `src/glove2h5.py` in a directory containing the GloVe vectors.

## Running Bias Tests

To run bias tests, run `src/main.py` with one or more tests and one or more models.  Note that each model may require additional command-line flags specifying locations of resources and other options. For example:

```
python src/main.py \
    -t weat1,weat2,weat3,weat4,sent-weat1,sent-weat2,sent-weat3,sent-weat4 \
    -m bow \
    --glove_path /export/b01/cmay/gbo/context-indep/glove.840B.300d.txt
```

Run `python src/main.py --help` to see a full list of options.

## Code Tests

```
flake8
```

## TODO

- track down NaNs in GenSen [Shikha]: looks like the denominator (stddev) of the effect size is 0 because a lot of the vectors are the same...possibly a problem with OOV, but all these words should be in GloVe (base word representations used). Can you make sure the vocab expansion method they implemented is being used?
- add options for concatenation of GenSen models [Shikha]
- implement SkipThought [Shikha]
- make sure BoW and GloVe results are the same for single-word WEAT tests [Shikha]
