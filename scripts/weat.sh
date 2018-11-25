#!/bin/bash

source user_config.sh

# debug
#python -m ipdb src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1 -m weat1 --exp_dir ${SAVE_DIR} --data_dir tests/ -s 1094 --n_samples 100 -m guse
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH}
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1 --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --elmo_combine concat

# BoW (consumes GloVe method)
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m bow --glove_path ${GLOVE_PATH}

# SkipThoughts

# InferSent
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m infersent --infersent_dir src/encoders --glove_path ${GLOVE_PATH}

# GenSen
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH}

# GUSE
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m guse 

# CoVe
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs cove_encs/

# ELMo
python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1 --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --elmo_combine concat
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --elmo_combine concat
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t sent_weat1,sent_weat2,sent_weat3,sent_weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --elmo_combine concat

# OpenAI GPT
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs openai_encs/
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t sent_weat1,sent_weat2,sent_weat3,sent_weat4 -m openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs encodings/openai/

# BERT
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m bert --bert_version base --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH}
#python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t sent_weat1,sent_weat2,sent_weat3,sent_weat4 -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH}
