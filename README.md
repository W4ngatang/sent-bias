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

#A minimum working example for running a weat test on glove and elmo:
Create user-specific paths `${EXP_DIR}` and `${GLOVE_PATH}` in `user_config.sh` and put that config file in the top level directory.
An example script to run things is in `scripts/weat.sh`.



## TODO

- add programmatic way of loading GloVe embeddings
- add support for other SentEval-included models: BoW, SkipThought, GenSen, Google USE (GUSE?)
- add support for jiant-included models: CoVe, OpenAI GPT
