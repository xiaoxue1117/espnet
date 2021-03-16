#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import epitran
import numpy as np
import gtn
import math
from collections import Counter

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--phoneme-dir", required=True, type=str)
    parser.add_argument("--phoneme-suf", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--input", required=True, type=str)
    return parser


args = get_parser().parse_args()



def get_phonemes(ph_units):
    phonemes = []
    phonemes.append("<blank>")
    with open(ph_units, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for li in lines:
            m, idx = li.split()
            phonemes.append(m)
    phonemes.append("<sos>")
    return phonemes


def main(args):
    langs = ["105", "106", "107", "302", "307", "402", "000"]

    priors = {}
    totals = {}
    lang_phonemes = {}
    for lid in langs:
        # load phonemes per lang
        lang_phonemes[lid] = get_phonemes(
            args.phoneme_dir + "train_swbd_ph_units_" + lid + args.phoneme_suf
        )
        priors[lid] = Counter()
        totals[lid] = 0

    with open(args.input, "r", encoding="utf-8") as tr_json:
        tr_data = json.load(tr_json)["utts"]
        for utt in tr_data.keys():
            entry = tr_data[utt]
            lid = entry["category"]
            phoneme_toks = entry["output"][1]["tokenid"].split(" ")
            for tok in phoneme_toks:
                priors[lid][int(tok)] += 1
                totals[lid] = totals[lid] + 1

    for lid in langs:
        print("---"+lid+"---")
        for tok in range(len(priors[lid].keys())):
            print(lang_phonemes[lid][tok], "{:.5%}".format(priors[lid][tok]))
            #print(lang_phonemes[lid][tok], "{:.5%}".format(priors[lid][tok] / totals[lid]))

if __name__ == "__main__":

    main(args)
