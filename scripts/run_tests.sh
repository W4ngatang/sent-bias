#!/bin/bash

source user_config.sh

TESTS=weat1,weat2,weat3,weat4

# debug
#python -m ipdb src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max
python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m infersent --infersent_dir src/encoders --glove_path ${GLOVE_PATH}

# BoW (consumes GloVe method)
#python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m bow --glove_path ${GLOVE_PATH}

# SkipThoughts

# InferSent
#python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m infersent --infersent_dir src/encoders --glove_path ${GLOVE_PATH}

# GenSen
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH}

# GUSE
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m guse 

# CoVe
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/

# ELMo
#python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --combine_method max --elmo_combine add

# OpenAI GPT
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs openai_encs/

# BERT
#python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max
