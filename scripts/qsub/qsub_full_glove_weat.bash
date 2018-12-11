#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N sentbias_full_glove_weat
#$ -t 1-10
#$ -l h_rt=144:00:00,num_proc=1,mem_free=2G,ram_free=2G

source ~/.bashrc
source activate sentbias
suffix=".$(date '+%Y%m%d%H%M%S').${JOB_ID}"
num_samples=`echo '10^9' | bc`
python sentbias/main.py -m bow -t weat$SGE_TASK_ID --log_file log$suffix --n_samples $num_samples --glove_path /export/b02/cmay14/sentbias/glove.840B.300d.txt --results_path results.tsv$suffix
