#!/bin/bash

source user_config.sh

python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1,weat2,weat3,weat4 -m bert --bert_version base --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH}
python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t sent_weat1,sent_weat2,sent_weat3,sent_weat4 -m bert --bert_version large --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH}
