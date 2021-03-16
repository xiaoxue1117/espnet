#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=3        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
seed=1
debugmode=1
dumpdir= #`mktemp -d /scratch/ml-asr-XXXX`   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_lm=false
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

nbpe=2000
bpemode=bpe

space=true

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.

langs="105 106 107 302 307 402 000" #000 is swbd
recog="105 106 107 302 307 402 000"

. utils/parse_options.sh || exit 1;
echo $dumpdir

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

#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#  echo "stage 0: Setting up individual languages"
#  ./local/setup_languages.sh --langs "${langs}" --recog "${recog}"
#  for x in ${train_dev} ${recog_set}; do
#	  sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
#  done
#fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#  echo "stage 1: Feature extraction"
#  fbankdir=fbank
#  # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
#  for x in ${train_dev} ${recog_set}; do
#      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
#          data/${x} exp/make_fbank/${x} ${fbankdir}
#      utils/fix_data_dir.sh data/${x}
#  done
#fi

#if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#  # speed pert
#  sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/train/wav.scp
#  utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
#  utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
#  utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
#  utils/combine_data.sh --extra-files utt2uniq data/train_sp data/temp1 data/temp2 data/temp3
#  rm -r data/temp1 data/temp2 data/temp3
#  fbankdir=fbank
#  steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
#      data/train_swbd exp/make_fbank/train_swbd ${fbankdir}
#  utils/fix_data_dir.sh data/train_swbd
#  utils/validate_data_dir.sh data/train_swbd
#
#  awk -v p="sp0.9-" '{printf("%s %s%s\n", $1, p, $1);}' data/train/utt2spk > data/train_swbd/utt_map
#  utils/apply_map.pl -f 1 data/train_swbd/utt_map <data/train/text >data/train_swbd/text
#  awk -v p="sp1.0-" '{printf("%s %s%s\n", $1, p, $1);}' data/train/utt2spk > data/train_swbd/utt_map
#  utils/apply_map.pl -f 1 data/train_swbd/utt_map <data/train/text >>data/train_swbd/text
#  awk -v p="sp1.1-" '{printf("%s %s%s\n", $1, p, $1);}' data/train/utt2spk > data/train_swbd/utt_map
#  utils/apply_map.pl -f 1 data/train_swbd/utt_map <data/train/text >>data/train_swbd/text
#fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    exit 1;
    utils/combine_data.sh --extra_files utt2uniq data/train_swbd data/train_sp /project/ocean/sdalmia/fall_2020/projects/espnet-maskaugment/egs/swbd/asr1/data/train_nodup
    utils/combine_data.sh --extra_files utt2uniq data/dev_swbd data/dev /project/ocean/sdalmia/fall_2020/projects/espnet-maskaugment/egs/swbd/asr1/data/train_dev
    
    ln -s /project/ocean/sdalmia/fall_2020/projects/espnet-maskaugment/egs/swbd/asr1/data/eval2000 data/eval_000
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    sed -i 's/\[laughter\]/<laughter>/g' data/train_swbd/text
    sed -i 's/\[noise\]/<noise>/g' data/train_swbd/text
    sed -i 's/\[vocalized-noise\]/<vocalized-noise>/g' data/train_swbd/text

    sed -i 's/\[laughter\]/<laughter>/g' data/dev_swbd/text
    sed -i 's/\[noise\]/<noise>/g' data/dev_swbd/text
    sed -i 's/\[vocalized-noise\]/<vocalized-noise>/g' data/dev_swbd/text

    sed -i 's/\[laughter\]/<laughter>/g' data/eval_000/text
    sed -i 's/\[noise\]/<noise>/g' data/eval_000/text
    sed -i 's/\[vocalized-noise\]/<vocalized-noise>/g' data/eval_000/text

    sed -i 's/<vocalized-noise>/<v-noise>/g' data/train_swbd/text
    sed -i 's/<vocalized-noise>/<v-noise>/g' data/dev_swbd/text
    sed -i 's/<vocalized-noise>/<v-noise>/g' data/eval_000/text
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/train_swbd data/train_swbd.trim
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/dev_swbd data/dev_swbd.trim
  rm -rf data/train_swbd
  rm -rf data/dev_swbd
  mv data/train_swbd.trim data/train_swbd
  mv data/dev_swbd.trim data/dev_swbd
  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
  utils/fix_data_dir.sh data/${train_set}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  dump.sh --cmd "$train_cmd" --nj 20 --do_delta ${do_delta} \
      data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
  dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
      data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
  for rtask in ${recog_set}; do
      feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
      dump.sh --cmd "$train_cmd" --nj 10 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
  done
fi

dict=data/lang_swbd/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_swbd/${train_set}_${bpemode}${nbpe}
nlsyms=data/lang_swbd/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_swbd/

    echo "make a non-linguistic symbol list"
    grep -e sp1.0 -e '^sw0' data/${train_set}/text | cut -f 2- -d' ' | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    #grep sp1.0 data/${train_set}/text | cut -f 2- -d' ' | tr " " "\n" | sort | uniq | grep "<" | grep -v "<unk>" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    offset=$(wc -l < ${dict})

    grep -e sp1.0 -e '^sw0' data/${train_set}/text | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_swbd/input.txt 
    spm_train --user_defined_symbols="$(tr "\n" "," < ${nlsyms})" --input=data/lang_swbd/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --unk_piece="<differentunk>"
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_swbd/input.txt | tr ' ' '\n' | sort | uniq | grep -v "<unk>" | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict} 
    wc -l ${dict}
fi

#dict_ph=data/lang_swbd/${train_set}_ph_units.txt
space_tag=
space_opt="--space"
if [ "$space" = false ] ; then
    space_tag=nosp
    space_opt=
fi
dict_ph=data/lang_swbd/${train_set}_ph_units_ipa${space_tag}.txt
bpemodel_ph=data/lang_swbd/${train_set}_ph

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "<unk> 1" > ${dict_ph} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat ${nlsyms} | grep -v "<unk>" | awk '{print $0 " " NR+1}' >> ${dict_ph}

    for lang in ${langs}; do
        if [ "$lang" = "000" ]; then
            grep '^sw0' data/${train_set}/text | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_swbd/input_${lang}.txt
        else
            grep sp1.0-${lang} data/${train_set}/text | cut -f 2- -d' ' | grep -v -e '^\s*$' > data/lang_swbd/input_${lang}.txt
        fi
        python local/learn_phonemes${space_tag}.py ${lang} "ph" ${nlsyms} data/lang_swbd/input_${lang}.txt | \
            sed "s/  */ /g" | \
            sed "s/^  *//g" | \
            sed "s/^<sp> //g" | \
            sed "s/ <sp>$//g" \
            > ${bpemodel_ph}_${lang}_ipa${space_tag}.model
        #cat data/lang_swbd/input_${lang}.txt | \
        #    #python utils/learn_phonemes.py ${lang} "ph" ${nlsyms} | \
        #    python local/learn_phonemes${space_tag}.py ${lang} "ph" ${nlsyms} | \
        #    sed "s/  */ /g" | \
        #    sed "s/^  *//g" | \
        #    sed "s/^<sp> //g" | \
        #    sed "s/ <sp>$//g" \
        #    > ${bpemodel_ph}_${lang}_ipa${space_tag}.model
        
        #lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}.txt
        lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}_ipa${space_tag}.txt
        echo "<unk> 1" > ${lang_dict_ph} # <unk> must be 1, 0 will be used for "blank" in CTC
        cat ${nlsyms} | grep -v "<unk>" | awk '{print $0 " " NR+1}' >> ${lang_dict_ph}
        offset=$(wc -l < ${lang_dict_ph})
        cat ${bpemodel_ph}_${lang}_ipa${space_tag}.model | tr ' ' '\n' | sort | uniq | grep -Ev "^$" | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${lang_dict_ph}
        wc -l ${lang_dict_ph}
    done
    
    [ -e ${bpemodel_ph}_ipa${space_tag}.model ] && rm ${bpemodel_ph}_ipa${space_tag}.model
    cat ${bpemodel_ph}_*_ipa${space_tag}.model > ${bpemodel_ph}_ipa${space_tag}.model
    #cat ${bpemodel_ph}_*.model > ${bpemodel_ph}.model
    #cat ${bpemodel_ph}.model | tr ' ' '\n' | sort | uniq | grep -Ev "^$" | awk '{print $0 " " NR+5}' >> ${dict_ph} 
    offset=$(wc -l < ${dict_ph})
    cat ${bpemodel_ph}_ipa${space_tag}.model | tr ' ' '\n' | sort | uniq | grep -Ev "^$" | awk -v offset=${offset} '{print $0 " " NR+offset}' >> ${dict_ph} 
    wc -l ${dict_ph}
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "make json files"
    data2json.sh --nj 16 --feat ${feat_tr_dir}/feats.scp --text data/${train_set}/text --bpecode ${bpemodel}.model \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "working on tr"
    cp ${feat_tr_dir}/data.json ${feat_tr_dir}/data_ph.json 
    
    for lang in ${langs}; do
        if [ "$lang" = "000" ]; then
            grep '^sw0' data/${train_set}/text > data/${train_set}/text_${lang}
        else
            grep \\-${lang}_ data/${train_set}/text > data/${train_set}/text_${lang}
        fi

        lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}_ipa${space_tag}.txt
        #lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}.txt
        local/update_ph_multi_parallel_json.py ${space_opt} --no-punct --lang ${lang} --input-json ${feat_tr_dir}/data_ph.json --output-json ${feat_tr_dir}/data_ph.json --units ${lang_dict_ph} --text-file data/${train_set}/text_${lang} --non-lang-file ${nlsyms}

    done

fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "working on dev test"
    cp ${feat_dt_dir}/data.json ${feat_dt_dir}/data_ph.json 
    
    for lang in ${langs}; do
        if [ "$lang" = "000" ]; then
            grep '^sw0' data/${train_dev}/text > data/${train_dev}/text_${lang}
        else
            grep ^${lang}_ data/${train_dev}/text > data/${train_dev}/text_${lang}
        fi

        lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}_ipa${space_tag}.txt
        #lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}.txt
        local/update_ph_multi_parallel_json.py ${space_opt} --no-punct --lang ${lang} --input-json ${feat_dt_dir}/data_ph.json --output-json ${feat_dt_dir}/data_ph.json --units ${lang_dict_ph} --text-file data/${train_dev}/text_${lang} --non-lang-file ${nlsyms}
    done

    for lang in ${langs}; do 
        rtask=eval_${lang}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}_ipa${space_tag}.txt
        #lang_dict_ph=data/lang_swbd/${train_set}_ph_units_${lang}.txt
        cp ${feat_recog_dir}/data.json ${feat_recog_dir}/data_ph.json 
        
        local/update_ph_multi_parallel_json.py ${space_opt} --no-punct --lang ${lang} --input-json ${feat_recog_dir}/data_ph.json --output-json ${feat_recog_dir}/data_ph.json --units ${lang_dict_ph} --text-file data/${rtask}/text --non-lang-file ${nlsyms}
    done
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    local/category.py --input-json ${feat_tr_dir}/data_ph.json --output-json ${feat_tr_dir}/data_ph_cat.json --sp 
    local/category.py --input-json ${feat_dt_dir}/data_ph.json --output-json ${feat_dt_dir}/data_ph_cat.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        local/category.py --input-json ${feat_recog_dir}/data_ph.json --output-json ${feat_recog_dir}/data_ph_cat.json
    done
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    #for lang in ${langs}; do
    #    if [ "$lang" = "105" ]; then
    #        lang_name="tur"
    #    elif [ "$lang" = "106" ]; then
    #        lang_name="tgl" 
    #    elif [ "$lang" = "107" ]; then
    #        lang_name="vie"
    #    fi
    #    allo_dir=local/allovera_json/
    #    local/prep_allo_map.py --map-json ${allo_dir}/${lang_name}.json --phoneme-txt data/lang_swbd/${train_set}_ph_units_${lang}_ipa${space_tag}.txt --phone-txt data/lang_swbd/${train_set}_ph_units_ipa${space_tag}.txt --output ${feat_tr_dir}/phonemap_${lang} --nlsyms-txt ${nlsyms}
    #done
    local/prep_allo_gtn_swbd.py --map-dir local/allovera_json/ --phoneme-dir data/lang_swbd/ \
        --phoneme-suf _ipa.txt --allphoneme-txt data/lang_swbd/${train_set}_ph_units_ipa${space_tag}.txt \
        --nlsyms-txt ${nlsyms} --output ${feat_tr_dir}/
fi

echo $dumpdir
