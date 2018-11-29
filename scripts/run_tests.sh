#!/bin/bash

source user_config.sh

TESTS=heilman_double_bind_ambiguous_1,heilman_double_bind_clear_1

# debug
#python -m ipdb src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH}

# BoW (consumes GloVe method)
python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m bow --glove_path ${GLOVE_PATH}

# SkipThoughts

# InferSent
#python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m infersent --infersent_dir src/encoders --glove_path ${GLOVE_PATH}

# GenSen
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH}

# GUSE
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m guse 

# CoVe
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/
#python src/main.py --log_file ${SAVE_DIR}/log.log -t sent-weat1,sent-weat2,sent-weat3,sent-weat4 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/

# ELMo
#python src/main.py --log_file ${SAVE_DIR}/log.log -t angry_black_woman_stereotype,angry_black_woman_stereotype_b --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --combine_method max --elmo_combine add

# OpenAI GPT
#python src/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs openai_encs/

# BERT
#python src/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version base --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max
