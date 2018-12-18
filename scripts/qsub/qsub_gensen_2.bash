#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N sentbias_gensen_2
#$ -l h_rt=24:00:00,num_proc=1,mem_free=16G,ram_free=16G

source ~/.bashrc
source activate sentbias
export TFHUB_CACHE_DIR=/export/b02/cmay14/sentbias/tfhub_cache
suffix=".$(date '+%Y%m%d%H%M%S').${JOB_ID}"
python sentbias/main.py --parametric -m gensen --use_cpu --log_file log$suffix --results_path results.tsv$suffix --glove_h5_path /export/b02/cmay14/sentbias/glove.840B.300d.h5 --gensen_dir /export/b02/cmay14/sentbias/gensen --gensen_version nli_large_bothskip
