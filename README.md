# sent-bias

`environment.yml` has all dependencies for Conda environment. Everything should
be installable with pip.

Example glove representations for weat tests are included.

For elmo:

```
cd scripts
./weat_glove2elmo.sh $WEATTESTNAME
./generate_elmo_embeddings.sh $WEATTESTNAME
# for example
./weat_glove2elmo.sh weat1
./generate_elmo_embeddings.sh weat1
```

A minimum working example for running a weat test on glove and elmo:

```
cd src
python run_weat.py
```
