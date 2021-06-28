lang=$1
exp=$2

python local/count_recurring_utts.py --input data/eval_${lang}/text --decode ${2}/decode_eval_${lang}_decode_maskedphones_model.last10_.avg_ignoresp/ph/data.json --units /scratch/ml-asr-4Izl/train_swbd/deltafalse/phone_units.txt --phonemes /scratch/ml-asr-4Izl/train_swbd/deltafalse/train_swbd_ph_units_${lang}_ipa.txt
