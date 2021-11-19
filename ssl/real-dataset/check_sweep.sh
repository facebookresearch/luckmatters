#!/bin/bash

logdir=$1
shift
run_analysis=$1
shift
match_file=$1
shift

if [ "$run_analysis" -eq "1" ]; then
  python ~/tools2/analyze.py --logdirs $logdir --log_regexpr_json ${match_file} --loader=log --num_process 1
fi

echo python ~/tools2/stats.py --logdirs $logdir --key_stats acc --descending --topk_mean 1 "$@"
python ~/tools2/stats.py --logdirs $logdir --key_stats acc --descending --topk_mean 1 --groups / "$@"
