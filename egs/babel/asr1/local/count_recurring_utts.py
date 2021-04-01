#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter
from itertools import groupby

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--decode", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    parser.add_argument("--phonemes", required=True, type=str)
    return parser

args = get_parser().parse_args()

def main(args):
    phoneme_units = ["<b>"]
    with open(args.phonemes, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split())==1:
                phone_units.append("")
            else:
                phoneme = line.strip().split()[0]
                phoneme_units.append(phoneme)

    units = ["<b>"]
    with open(args.units, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split())==1:
                units.append("")
            else:
                phone = line.strip().split()[0]
                units.append(phone)

    utt_counts = Counter()
    utt_accents = {}
    utt_dict = {}
    utt_lexicon = {}
    utt_phonemes = {}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            #uttid, utttext = line.split('\t')
            uttid, utttext = line.strip().split(' ', maxsplit=1)
            uttaccent = uttid[:9]
            utt_counts[utttext] += 1
            if utttext in utt_accents.keys():
                utt_accents[utttext][uttaccent] += 1
            else:
                utt_accents[utttext] = Counter()
                utt_accents[utttext][uttaccent] += 1

        common = utt_counts.most_common(10)
        #print(common)
        common_utts = []
        for utt, count in common:
            #print(utt, utt_accents[utt])
            common_utts.append(utt)

        for utt in common_utts:
            utt_dict[utt] = []
            utt_lexicon[utt] = []
        for line in lines:
            uttid, utttext = line.strip().split(' ', maxsplit=1)
            if utttext in common_utts:
                utt_dict[utttext].append(uttid)

    for text in common_utts:
        count = 0
        with open(args.decode, "r", encoding="utf-8") as f:
            data = json.load(f)["utts"]
            for utt in data.keys():
                if utt in utt_dict[text]:
                    res = data[utt]["output"][1]
                    if res["rec_token"] == res["token"]:
                        count += 1
                        phones = data[utt]["output"][2]["phones"]
                        collapsed_indices = [x[0] for x in groupby(phones)] 
                        collapsed_indices = [units[x] for x in collapsed_indices if x != 0]
                        pron = "".join(collapsed_indices)
                        utt_lexicon[text].append(pron)
                        if utt not in utt_phonemes.keys():
                            phonemes = "".join([phoneme_units[int(c)] for c in res["token"].split()])
                            utt_phonemes[text] = phonemes
        #print(text, count, utt_counts[text])

    print("TEXT", "PHONEMES", "PHONES", sep="\t")
    for text in utt_phonemes.keys():
        pron_list = utt_lexicon[text]
        pron_set = set(pron_list)
        res = [(p, pron_list.count(p)) for p in pron_set]
        res.sort(key=lambda x:-x[1])
        print(text, utt_phonemes[text], res, sep="\t")

if __name__ == "__main__":

    main(args)
