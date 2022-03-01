#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


nj=20
train_set=train_nodev
valid_set=dev100
test_set=dev

asr_config=conf/train_asr_transformer_specaug.yaml
asr_config=conf/alpha0.yaml
asr_config=conf/moe_freeze_ft.yaml

lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml


# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="1.1 0.9 1.0"
# --bpe_char_cover 0.995 \
# check that all languages have the same nlsyms 

./asr.sh \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_type bpe \
    --nbpe 50 \
    --feats_type raw \
    --bpe_train_text "data/${train_set}/text" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "valid.acc.best.pth" \
    --feats_normalize "utterance_mvn" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --nj "${nj}" \
    --inference_nj "${nj}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --expdir exp_MOE \
    --ngpu 1 \
    --lm_train_text "data/${train_set}/text" "$@"

 # sbatch -t 0 --exclude tir-0-17,tir-0-15,tir-0-36,tir-0-11  --cpus-per-task=2 --gres=gpu:TITANX:1  --mem=20G  run.sh --stage 11  --stop_stage 11 --asr_tag baseline

 # sbatch -t 0 --exclude tir-0-17,tir-0-15,tir-0-36,tir-0-11  --cpus-per-task=20  --mem=120G  run.sh --stage 12  --stop_stage 13 --asr_tag baseline
