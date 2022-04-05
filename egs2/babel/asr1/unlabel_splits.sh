#!/bin/bash

mkdir old_splits

idxs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13")
for i in "${idxs[@]}"; do
    file=exp_semi_supervised/asr_stats_raw_107_bpe1000_sp/splits14/text/split.${i}
    cp ${file} old_splits
    cut -d ' ' -f 1 ${file} > aux
    awk '{print $0" semi"}' aux > ${file}
    rm aux  
done

