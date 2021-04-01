#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

dir=${1}
typ=${2}

if [ "$typ" == "wer" ]; then
    wer=".wrd"
    langs="105 106 107 302 307 402 000"
    
    for lang in ${langs}; do
        if [ "${lang}" = "000" ]; then
            grep "(sw0" ${dir}/ref${wer}.trn > ${dir}/ref_${lang}${wer}.trn
            grep "(sw0" ${dir}/hyp${wer}.trn > ${dir}/hyp_${lang}${wer}.trn
        else
            grep "(${lang}_" ${dir}/ref${wer}.trn > ${dir}/ref_${lang}${wer}.trn
            grep "(${lang}_" ${dir}/hyp${wer}.trn > ${dir}/hyp_${lang}${wer}.trn
        fi
        sclite -r ${dir}/ref_${lang}${wer}.trn trn -h ${dir}/hyp_${lang}${wer}.trn trn -i rm -o dtl stdout > ${dir}/dtl_${lang}${wer}.txt
    done
    
    grep "Percent Total Error"  ${dir}/dtl_*${wer}.txt
else
    wer=""
    dir=${dir}/ph/
    langs="105 106 107 302 307 402 000"
    
    for lang in ${langs}; do
        if [ "${lang}" = "000" ]; then
            grep "(sw0" ${dir}/ref${wer}.trn > ${dir}/ref_${lang}${wer}.trn
            grep "(sw0" ${dir}/hyp${wer}.trn > ${dir}/hyp_${lang}${wer}.trn
        else
            grep "(${lang}_" ${dir}/ref${wer}.trn > ${dir}/ref_${lang}${wer}.trn
            grep "(${lang}_" ${dir}/hyp${wer}.trn > ${dir}/hyp_${lang}${wer}.trn
        fi
        sclite -r ${dir}/ref_${lang}${wer}.trn trn -h ${dir}/hyp_${lang}${wer}.trn trn -i rm -o dtl stdout > ${dir}/dtl_${lang}${wer}.txt
    done
    
    grep "Percent Total Error"  ${dir}/dtl_*${wer}.txt
fi
