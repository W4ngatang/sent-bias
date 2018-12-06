#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N sentbias_ulmfit_2
#$ -l h_rt=24:00:00,num_proc=1,mem_free=16G,ram_free=16G

source ~/.bashrc
source activate sentbias
export TFHUB_CACHE_DIR=/export/b02/cmay14/sentbias/tfhub_cache
suffix=".$(date '+%Y%m%d%H%M%S').${JOB_ID}"
python sentbias/main.py -m ulmfit --use_cpu --log_file sent-word.log$suffix --results_path results.sent-word.tsv$suffix --ulmfit_dir /export/b02/cmay14/sentbias/ulmfit --layer_combine_method last
