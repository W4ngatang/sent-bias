#!/bin/bash

source user_config.sh

echo 'Note: this script should be called from the root of the repository' >&2

TESTS=weat1,weat2,weat3,weat3b,weat4,weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10,sent-weat1,sent-weat2,sent-weat3,sent-weat3b,sent-weat4,sent-weat5,sent-weat5b,sent-weat6,sent-weat6b,sent-weat7,sent-weat7b,sent-weat8,sent-weat8b,sent-weat9,sent-weat10,angry_black_woman_stereotype,angry_black_woman_stereotype_b,sent-angry_black_woman_stereotype,sent-angry_black_woman_stereotype_b,heilman_double_bind_competent_1,heilman_double_bind_competent_1-,heilman_double_bind_competent_one_sentence,heilman_double_bind_competent_one_word,sent-heilman_double_bind_competent_one_word,heilman_double_bind_likable_1,heilman_double_bind_likable_1-,heilman_double_bind_likable_one_sentence,heilman_double_bind_likable_one_word,sent-heilman_double_bind_likable_one_word
#TESTS=weat1
set -e

SEED=1111

# debug
#python ipdb sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} --combine_method max 

# BoW (consumes GloVe method)
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m bow --glove_path ${GLOVE_PATH} -s ${SEED} --ignore_cached_encs

# SkipThoughts

# InferSent
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m infersent --infersent_dir sentbias/encoders --glove_path ${GLOVE_PATH} -s ${SEED} --ignore_cached_encs

# GenSen
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 --exp_dir ${SAVE_DIR} --data_dir tests/ -m gensen --gensen_dir ${GENSEN_DIR} --glove_path ${GLOVE_PATH} -s ${SEED} --ignore_cached_encs

# GUSE
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m guse -s ${SEED} --ignore_cached_encs

# CoVe
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m cove --exp_dir ${SAVE_DIR} --data_dir tests/ --cove_encs encodings/cove/ -s ${SEED} --ignore_cached_encs

# ELMo
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --combine_method max --elmo_combine add -s ${SEED} --ignore_cached_encs
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} --exp_dir ${SAVE_DIR} --data_dir tests/ -m elmo --combine_method max --elmo_combine concat -s ${SEED} --ignore_cached_encs

# OpenAI GPT
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m openai --exp_dir ${SAVE_DIR} --data_dir tests/ --openai_encs encodings/openai/ -s ${SEED} --results_path openai.tsv --dont_cache_encs --ignore_cached_encs

# BERT
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} -s ${SEED} --ignore_cached_encs
#python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m bert --bert_version base --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH} -s ${SEED} --ignore_cached_encs

