#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N sentbias_elmo_1
#$ -l h_rt=24:00:00,num_proc=1,mem_free=16G,ram_free=16G

SENTBIAS_ROOT=/export/b02/cmay14/sentbias

source ~/.bashrc
source activate sentbias
export TFHUB_CACHE_DIR=$SENTBIAS_ROOT/tfhub_cache
export ALLENNLP_CACHE_ROOT=$SENTBIAS_ROOT/allennlp_cache
export PYTORCH_PRETRAINED_BERT_CACHE=$SENTBIAS_ROOT/bert_cache
suffix=".$(date '+%Y%m%d%H%M%S').${JOB_ID}"
python sentbias/main.py -m elmo --use_cpu --log_file log$suffix --results_path results.tsv$suffix --layer_combine_method add --time_combine_method max
