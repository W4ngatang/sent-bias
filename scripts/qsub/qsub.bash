#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N sentbias
#$ -l h_rt=144:00:00,num_proc=1,mem_free=16G,ram_free=16G

source ~/.bashrc
source activate sentbias
export TFHUB_CACHE_DIR=/export/b02/cmay14/sentbias/tfhub_cache
suffix=".$(date '+%Y%m%d%H%M%S').${JOB_ID}"
python sentbias/main.py -m bow --use_cpu --log_file log$suffix --results_path results.tsv$suffix --glove_path /export/b02/cmay14/sentbias/glove.840B.300d.txt
python sentbias/main.py -m elmo --use_cpu --log_file log$suffix --results_path results.tsv$suffix
python sentbias/main.py -m elmo --use_cpu --log_file log$suffix --results_path results.tsv$suffix --layer_combine_method concat
python sentbias/main.py -m elmo --use_cpu --log_file log$suffix --results_path results.tsv$suffix --time_combine_method mean
python sentbias/main.py -m bert --use_cpu --log_file log$suffix --results_path results.tsv$suffix
python sentbias/main.py -m bert --use_cpu --log_file log$suffix --results_path results.tsv$suffix --bert_version large
python sentbias/main.py -m gensen --use_cpu --log_file log$suffix --results_path results.tsv$suffix --glove_h5_path /export/b02/cmay14/sentbias/glove.840B.300d.h5 --gensen_dir /export/b02/cmay14/sentbias/gensen
python sentbias/main.py -m gensen --use_cpu --log_file log$suffix --results_path results.tsv$suffix --glove_h5_path /export/b02/cmay14/sentbias/glove.840B.300d.h5 --gensen_dir /export/b02/cmay14/sentbias/gensen --gensen_version nli_large_bothskip
python sentbias/main.py -m infersent --use_cpu --log_file log$suffix --results_path results.tsv$suffix --glove_path /export/b02/cmay14/sentbias/glove.840B.300d.txt --infersent_dir /export/b02/cmay14/sentbias/infersent
python sentbias/main.py -m guse --use_cpu --log_file log$suffix --results_path results.tsv$suffix
python sentbias/main.py -m openai --use_cpu --log_file log$suffix --results_path results.tsv$suffix --openai_encs /export/b02/cmay14/sentbias/openai
