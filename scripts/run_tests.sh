#!/bin/bash

source user_config.sh

echo 'Note: this script should be called from the root of the repository' >&2

TESTS=weat1,weat2,weat3,weat3b,weat4,weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10,sent-weat1,sent-weat2,sent-weat3,sent-weat4,angry_black_woman_stereotype,angry_black_woman_stereotype_b,heilman_double_bind_ambiguous_1,heilman_double_bind_clear_1,heilman_double_bind_ambiguous_1+3-,heilman_double_bind_clear_1+3-
#TESTS=weat1,weat2,weat3,weat4
set -e

SEED=2222

# debug
#python ipdb sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max 

# BoW (consumes GloVe method)
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ bow --glove_path ${GLOVE_PATH} -s ${SEED}

# SkipThoughts

# InferSent
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ infersent --infersent_dir sentbias/encoders --glove_path ${GLOVE_PATH} -s ${SEED}

# GenSen
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH} -s ${SEED}

# GUSE
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ guse -s ${SEED}

# CoVe
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/ -s ${SEED}

# ELMo
python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ elmo --combine_method max --elmo_combine add -s ${SEED}

# OpenAI GPT
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs openai_encs/ -s ${SEED}

# BERT
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max -s ${SEED}
