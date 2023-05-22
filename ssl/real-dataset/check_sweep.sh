#!/bin/bash

logdir=$1
shift
run_analysis=$1
shift

if [ "$run_analysis" -eq "1" ]; then
  python ~/tools2/analyze.py $logdir --num_process 1 
fi

echo python ~/tools2/stats.py $logdir 
python ~/tools2/stats.py $logdir --groups / "$@"
