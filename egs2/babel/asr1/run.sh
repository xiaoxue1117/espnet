#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="107"
recog=$lang


nj=10
train_set=train_${lang}
valid_set=dev_${lang}
test_set=eval_${lang}


#BASELINE CONFIG : 
asr_config=conf/tuning/train_asr_hubert_base.yaml

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
    --inference_asr_model "valid.acc.best.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" \
    --nj "${nj}" \
    --inference_nj "${nj}" \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --ngpu 1 \
    --lang ${lang} \
    --expdir exp_semi_supervised \
    --dumpdir dump/dump_lid_${lang} \
    --lm_train_text "data/${train_set}/text" "$@" \
    --num_splits_asr 14 \


# sbatch -t 0 --exclude tir-0-17,tir-0-15,tir-0-36,tir-0-11  --cpus-per-task=4 --mem=30G   run.sh --stage 1  --stop_stage 5

# sbatch -t 0 --exclude tir-0-17,tir-0-15,tir-0-36,tir-0-11  --cpus-per-task=10 --mem=40G   run.sh --stage 10  --stop_stage 10 
