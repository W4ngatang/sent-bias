# sent-bias

## Setup 

`environment.yml` has all dependencies for Conda environment. Everything should be installable with pip.
Create the environment with `conda env create -f environment.yml`.
Activate the environment with `source activate sentbias`.

You will also need to download pretrained model weights for each model
you want to test.  Instructions for each supported model are as
follows.

### GloVe 

Example glove representations for weat tests are included in `tests/`.

### ELMo

```
cd scripts
./weat_glove2elmo.sh $WEATTESTNAME
./generate_elmo_embeddings.sh $WEATTESTNAME
# for example
./weat_glove2elmo.sh weat1
./generate_elmo_embeddings.sh weat1
```

### Infersent

Download the model checkpoints from the [original repo](https://github.com/facebookresearch/InferSent) and put them in `src/encoders`.

### GenSen

Download the model checkpoints.
Note, you need to process your GloVe word vectors into an HDF5 format. Run `src/glove2h5.py` in a directory containing the GloVe vectors.


## Running Bias Tests

Create user-specific paths `${EXP_DIR}` and `${GLOVE_PATH}` in `user_config.sh`, e.g.: 

```
EXP_DIR=path/to/save/stuff
GLOVE_PATH=path/to/glove/vectors
```

Put that config file in the top level directory.

An example script to run things is in `scripts/weat.sh`. To change the test, change the `-t` flag. To change the model, change the `-m` flag (currently accepted: InferSent, GloVe, ELMo)..



## TODO

- track down NaNs in GenSen [Shikha]: looks like the denominator (stddev) of the effect size is 0 because a lot of the vectors are the same...possibly a problem with OOV, but all these words should be in GloVe (base word representations used). Can you make sure the vocab expansion method they implemented is being used?
- add options for concatenation of GenSen models [Shikha]
- implement SkipThought [Shikha]
- make sure BoW and GloVe results are the same for single-word WEAT tests [Shikha]
