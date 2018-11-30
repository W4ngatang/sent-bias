#!/bin/bash

set -e

#TESTS=weat1,weat2,weat3,weat3b,weat4,weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10,sent-weat1,sent-weat2,sent-weat3,sent-weat4,angry_black_woman_stereotype,angry_black_woman_stereotype_b,heilman_double_bind_ambiguous_1,heilman_double_bind_clear_1,heilman_double_bind_ambiguous_1+3-,heilman_double_bind_clear_1+3-
TESTS=weat1
SAVE_DIR=output

source user_config.sh

# debug
#python -m ipdb sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max

# BoW (consumes GloVe method)
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m bow --glove_path ${GLOVE_PATH}

# SkipThoughts

# InferSent
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m infersent --infersent_dir sentbias/encoders --glove_path ${GLOVE_PATH}

# GenSen
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH}

# GUSE
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m guse 

# CoVe
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/

# ELMo
python -m ipdb sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --combine_method max --elmo_combine add

# OpenAI GPT
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs openai_encs/

# BERT
#python -m sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max
