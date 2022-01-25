#!/bin/bash

logdir=$1
shift
run_analysis=$1
shift

if [ "$run_analysis" -eq "1" ]; then
  python ~/tools2/analyze.py $logdir --num_process 32 --result_group "*" 
fi

echo python ~/tools2/stats.py $logdir --key_stats $key_stat --groups / "$@"
python ~/tools2/stats.py $logdir --groups / "$@"
