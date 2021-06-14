#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test"

lm_config=conf/lm_transformer.yaml
inference_config=conf/decode_asr.yaml

## APC
# opts="--expdir exp_s3prl_apc"
# asr_conf=train_asr_conformer_s3prl_frontend_apc_1_1; opts=${opts:-}" --feats_normalize null"

## CPC
opts="--expdir exp_s3prl_cpc"
asr_conf=train_asr_conformer_s3prl_frontend_cpc_1_1; opts=${opts:-}" --feats_normalize null"

## Wav2vec2
# opts="--expdir exp_s3prl_wav2vec2"
# asr_conf=train_asr_conformer_s3prl_frontend_wav2vec2_1_1; opts=${opts:-}" --feats_normalize null"

## Hubert
# opts="--expdir exp_s3prl_hubert"
# asr_conf=train_asr_conformer_s3prl_frontend_hubert_1_1; opts=${opts:-}" --feats_normalize null"

./asr.sh \
    --stage 10 --stop-stage 10 \
    --lang en \
    --ngpu 1 \
    --nbpe 500 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "conf/tuning_s3prl/${asr_conf}.yaml" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/${train_set}/text" "$@" ${opts}
