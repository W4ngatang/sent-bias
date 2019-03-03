# This is a list of partial commands to run to regenerate all results
# in the paper.  It's designed to be used with GNU Parallel (or xargs):
#
# sed '/^#/d' scripts/tasks.sh | shuf | parallel --bar --line-buffer -j 2 \
#     {} \
#     --use_cpu \
#     --glove_path /media/cjmay/Data1/sent-bias/glove.840B.300d.txt \
#     --glove_h5_path /media/cjmay/Data1/sent-bias/glove.840B.300d.h5 \
#     --infersent_dir /media/cjmay/Data1/sent-bias/infersent \
#     --gensen_dir /media/cjmay/Data1/sent-bias/gensen \
#     --openai_encs /media/cjmay/Data1/sent-bias/ckpts/jiant/sentbias-openai/openai
#
# That command strips the commented-out lines from this file, shuffles
# the remaining lines, and runs them in parallel (two at a time),
# passing six additional command-line flags to each one.
python sentbias/main.py -m bert --log_file log.bert-1 --results_path results.tsv.bert-1 --bert_version bert-base-uncased
python sentbias/main.py -m bert --log_file log.bert-2 --results_path results.tsv.bert-2 --bert_version bert-large-uncased
python sentbias/main.py -m bert --log_file log.bert-3 --results_path results.tsv.bert-3 --bert_version bert-large-cased
python sentbias/main.py -m bert --log_file log.bert-4 --results_path results.tsv.bert-4 --bert_version bert-base-cased
python sentbias/main.py -m bow --log_file log.bow --results_path results.tsv.bow
python sentbias/main.py -m elmo --log_file log.elmo-1 --results_path results.tsv.elmo-1 --layer_combine_method add --time_combine_method max
python sentbias/main.py -m elmo --log_file log.elmo-2 --results_path results.tsv.elmo-2 --layer_combine_method concat --time_combine_method max
python sentbias/main.py -m elmo --log_file log.elmo-3 --results_path results.tsv.elmo-3 --layer_combine_method add --time_combine_method mean
python sentbias/main.py -m gensen --log_file log.gensen-1 --results_path results.tsv.gensen-1 --gensen_version nli_large_bothskip_parse
python sentbias/main.py -m gensen --log_file log.gensen-2 --results_path results.tsv.gensen-2 --gensen_version nli_large_bothskip
python sentbias/main.py -m gensen --log_file log.gensen-3 --results_path results.tsv.gensen-3 --gensen_version nli_large_bothskip_parse,nli_large_bothskip
python sentbias/main.py -m guse --log_file log.guse --results_path results.tsv.guse
python sentbias/main.py -m infersent --log_file log.infersent --results_path results.tsv.infersent
python sentbias/main.py -m openai --log_file log.openai --results_path results.tsv.openai
