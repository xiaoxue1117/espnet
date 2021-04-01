#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir=
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml
lm_config=lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

bpemode=bpe
nbpe=2000

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10                  # the number of ST models to be averaged
use_snapshot_range=false                                                                                                                              snapshot_lower=
snapshot_upper=

# exp tag
tag="" # tag for managing experiments.

langs="105 106 107 302 307 402 000"
recog="105 106 107 302 307 402 000"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Train Directories
train_set=train_swbd
train_dev=dev_swbd

# LM Directories
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
lm_train_set=data/local/train.txt
lm_valid_set=data/local/dev.txt

recog_set=""
for l in ${recog}; do
  recog_set="eval_${l} ${recog_set}"
done
recog_set=${recog_set%% }


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=data/lang_swbd/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_swbd/${train_set}_${bpemode}${nbpe}
nlsyms=data/lang_swbd/non_lang_syms.txt

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})_ngpu${ngpu}
    #expname=${train_set}_${backend}_$(basename ${train_config%.*})_ngpu${ngpu}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})_ngpu${ngpu}_${tag}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Network Training"
    
    #asr_train.py \
    #--config ${train_config} \
    #--n-iter-process 8 \
    #--preprocess-conf ${preprocess_config} \
    #--ngpu 1 \
    #--backend ${backend} \
    #--outdir ${expdir}/results \
    #--tensorboard-dir tensorboard/${expname} \
    #--debugmode ${debugmode} \
    #--dict ${dict} \
    #--debugdir ${expdir} \
    #--minibatches ${N} \
    #--verbose ${verbose} \
    #--resume ${resume} \
    #--seed ${seed} \
    #--train-json ${feat_tr_dir}/data_ph_cat.json \
    #--valid-json ${feat_dt_dir}/data_ph_cat.json \
    #--phonemap-np ${feat_tr_dir}/phonemap_ \
    #--phonegraph ${feat_tr_dir}/phonegraph_
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --n-iter-process 8 \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data_ph_cat.json \
        --valid-json ${feat_dt_dir}/data_ph_cat.json \
        --phonemap-np ${feat_tr_dir}/phonemap_ \
        --phonegraph ${feat_tr_dir}/phonegraph_
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Averaging Models"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ST models
        recog_model=model.last${n_average}_${snapshot_upper}.avg
        opt="--log"
        
        if ${use_snapshot_range}; then
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots $(seq -f "${expdir}/results/snapshot.ep.%02g" ${snapshot_lower} ${snapshot_upper}) \
                --out ${expdir}/results/${recog_model} \
                --num ${n_average}
        else
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${expdir}/results/snapshot.ep.* \
                --out ${expdir}/results/${recog_model} \
                --num ${n_average}
        fi
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=15

    #extra_opts=""
    #if ${use_lm}; then
    #  extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    #fi
    
    recog_model=model.last${n_average}_${snapshot_upper}.avg
    echo "${recog_model}"

    pids=() # initialize pids
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${recog_model}
        #if ${use_lm}; then
        #    decode_dir=${decode_dir}_rnnlm_${lmtag}
        #fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        #feat_recog_dir=${dumpdir}/${rtask}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_ph_cat.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --batchsize 0 \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_ph_cat.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_ph_cat.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            #${extra_opts}

        local/score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpemodel}.model --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        local/score_sclite_ph.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir}/ph
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Decoding"
    nj=15

    #extra_opts=""
    #if ${use_lm}; then
    #  extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    #fi
    
    recog_model=model.last${n_average}_${snapshot_upper}.avg
    echo "${recog_model}"

    train_dev=tus2
    pids=() # initialize pids
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${recog_model}
        #if ${use_lm}; then
        #    decode_dir=${decode_dir}_rnnlm_${lmtag}
        #fi
        feat_recog_dir=${dumpdir}/${rtask}/

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_ph_cat.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --batchsize 0 \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_ph_cat.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_ph_cat.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            #${extra_opts}

        local/score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpemodel}.model --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        local/score_sclite_ph_unseen.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir}/ph
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 6: Decoding"
    nj=15

    #extra_opts=""
    #if ${use_lm}; then
    #  extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    #fi
    
    recog_model=model.last${n_average}_${snapshot_upper}.avg
    echo "${recog_model}"

    train_dev=inu  #uses the train_swbd cmvn instead of inu
    pids=() # initialize pids
    for rtask in ${train_dev}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${recog_model}
        #if ${use_lm}; then
        #    decode_dir=${decode_dir}_rnnlm_${lmtag}
        #fi
        feat_recog_dir=${dumpdir}/${rtask}/

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_ph_cat.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --batchsize 0 \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_ph_cat.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_ph_cat.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            #${extra_opts}

        local/score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpemodel}.model --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        local/score_sclite_ph_unseen.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir}/ph
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Decoding Test"
    nj=15

    #extra_opts=""
    #if ${use_lm}; then
    #  extra_opts="--rnnlm ${lmexpdir}/rnnlm.model.best ${extra_opts}"
    #fi
    
    recog_model=model.last${n_average}_${snapshot_upper}.avg
    echo "${recog_model}"

    #pids=() # initialize pids
    for rtask in ${recog_set}; do
    #(
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${recog_model}
        #if ${use_lm}; then
        #    decode_dir=${decode_dir}_rnnlm_${lmtag}
        #fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_ph_cat.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --batchsize 0 \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_ph_cat.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_ph_cat.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            #${extra_opts}

        local/score_sclite.sh --wer true --bpe ${nbpe} --bpemodel ${bpemodel}.model --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}
        local/score_sclite_ph.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir}/ph

        cp -r ${expdir}/${decode_dir} ${expdir}/${decode_dir}_ignoresp
        local/score_sclite_ph_ignoresp.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir}_ignoresp/ph 
    #) &
    #pids+=($!) # store background pids
    done
    #i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    #[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
