#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="107"
recog=$lang


nj=20
train_set=train_${lang}_1h
valid_set=dev_${lang}
test_set=eval_${lang}_small

#BASELINE CONFIG : 
asr_config=conf/tuning/train_asr_hubert_base_ss.yaml


lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml
nlsyms_txt=data/nlsym.txt

# speed perturbation related
speed_perturb_factors="1.1 0.9 1.0"


./asr.sh \
    --audio_format "flac.ark" \
    --lang noinfo \
    --local_data_opts "--langs ${lang} --recog ${recog}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --nbpe 1000 \
    --feats_normalize "utterance_mvn" \
    --feats_type raw \
    --nlsyms_txt "${nlsyms_txt}" \
    --bpe_nlsyms "$(perl -pe 's/\n/,/' data/nlsym.txt)" \
    --bpe_train_text "data/${train_set}/text" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --inference_asr_model "latest.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --nj "${nj}" \
    --inference_nj "${nj}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --ngpu 1 \
    --lang ${lang} \
    --expdir exp\
    --dumpdir dump/dump_lid_${lang}\
    --lm_train_text "data/${train_set}/text" "$@"



# sbatch -t 0 --exclude  tir-0-3,tir-1-23,tir-1-7,tir-1-11 --cpus-per-task=22  --mem=12G  --output=OUTPUTS/102_10_char.out  run_hubert.sh --stage 10  --stop_stage 10 

# sbatch -t 0 --exclude  tir-0-17,tir-0-3,tir-1-7 --cpus-per-task=2  --gres=gpu:TITANX:1 --mem=20G  run.sh --stage 11  --stop_stage 11 --asr_tag baseline_hubert

# sbatch -t 0 --exclude tir-0-3,tir-1-23,tir-1-7,tir-1-11  --cpus-per-task=20  --mem=120G  run_hubert.sh --stage 12  --stop_stage 13 --asr_tag continue_PT_iter2