#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev_other"
test_sets="test_clean test_other dev_clean dev_other"


./asr.sh \
    --skip_data_prep false \
    --skip_train false \
    --pretrained_model "/projects/tir6/general/dberrebb/hubertEE/espnet/egs2/librispeech_100/asr1/hubert_large_ft_ls960.espnet.pth" \
    --ignore_init_mismatch "true" \
    --skip_eval false \
    --feats_normalize "utterance_mvn" \
    --lang en \
    --ngpu 1 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --inference_asr_model "latest.pth" \
    --nj 10 \
    --use_lm false \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"


# ./run.sh --stage 10 --stop_stage 10 --train_set dev_other_small --valid_set dev_other_small --test_sets dev_other_small --token_type char --expdir exp_decoding --asr_config conf/hubertEE_FT_heads.yaml --nj 10

# ./run.sh --stage 11 --stop_stage 11  --token_type char --expdir exp_decoding --asr_config conf/hubertEE_FT_heads.yaml --asr_tag debug1

# ./run.sh --stage 12 --stop_stage 13 --test_sets dev_other_small --token_type char --expdir exp_decoding  --inference_nj 10 --asr_tag heads_5_10_15_20

# sbatch -t 0  --cpus-per-task=2  --mem=30G --gres=gpu:2080Ti:1 run.sh --stage 11 --stop_stage 11  --token_type char --expdir exp_baselines2 --asr_config conf/hubertEE_FT_heads.yaml --asr_tag heads_5_10_15_20



# sbatch -t 0  --cpus-per-task=10  --mem=120G --exclude tir-0-32,tir-1-32,tir-1-28,tir-0-3 run.sh --stage 1 --stop_stage 5 --train_set "train_fr" --valid_set "dev_fr" --test_sets "dev_fr test_as" --lang "fr" --local_data_opts "--lang fr" --audio_format "flac" --nj 10 --lm_train_text "data/train_fr/text"  --token_type char