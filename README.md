# sent-bias

## Setup 

`environment.yml` has all dependencies for Conda environment. Everything should be installable with pip.
Create the environment with `conda environment -f environment.yml`.
Activate the environment with `source activate sentbias`.

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



## Running Stuff

Create user-specific paths `${EXP_DIR}` and `${GLOVE_PATH}` in `user_config.sh`, e.g.: 

```
EXP_DIR=path/to/save/stuff
GLOVE_PATH=path/to/glove/vectors
```

Put that config file in the top level directory.

An example script to run things is in `scripts/weat.sh`. To change the test, change the `-t` flag. To change the model, change the `-m` flag (currently accepted: InferSent, GloVe, ELMo)..



## TODO

- track down NaNs in GenSen [Shikha]
- add options for concatenation of GenSen models [Shikha]
- implement SkipThought [Shikha]
- make sure BoW and GloVe results are the same for single-word WEAT tests [Shikha]
