#!/bin/bash

source user_config.sh

python src/run_weat.py --log_file ${SAVE_DIR}/log.log -t weat1 -m elmo --exp_dir ${SAVE_DIR} --data_dir tests/ --glove_path ${GLOVE_PATH}
