#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter
from itertools import groupby

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--decode", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    parser.add_argument("--phonemes", required=True, type=str)
    parser.add_argument("--tgt-phoneme", required=True, type=int)
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

    tgt_phoneme = args.tgt_phoneme
    phoneme_count = 0
    pre_counts = Counter()
    post_counts = Counter()
    phone_counts = Counter()
    preds = []
    with open(args.decode, "r", encoding="utf-8") as f:
        data = json.load(f)["utts"]
        for utt in data.keys():
            hyp = data[utt]["output"][1]["rec_token"].split().count(str(tgt_phoneme))
            ref = data[utt]["output"][1]["token"].split().count(str(tgt_phoneme))
            if hyp == ref:
                phones = data[utt]["output"][2]["phones"]
                phonemes = data[utt]["output"][2]["phonemes"]
                indices = [i for i, x in enumerate(phonemes) if x == tgt_phoneme]
                for i in indices:
                    phoneme_count += 1
                    #get phone
                    tgt_phone = phones[i]
                    phone_counts[units[tgt_phone]] += 1
                    #find pre-context
                    j = i - 1
                    pre = -1
                    while j >= 0:
                        if phones[j] != tgt_phone and phones[j] != 0:
                            pre_counts[units[phones[j]]] += 1
                            pre = phones[j]
                            break
                        else:
                            j -= 1
                    #find post-context
                    j = i + 1
                    post = len(phones)
                    while j < len(phones):
                        if phones[j] != tgt_phone and phones[j] != 0:
                            post_counts[units[phones[j]]] += 1
                            post = phones[j]
                            break
                        else:
                            j += 1
                    #save preds
                    preds.append((tgt_phone, pre, post))

    print("Phoneme:", phoneme_units[tgt_phoneme], phoneme_count)
    #print(pre_counts.most_common(10))
    #print(post_counts.most_common(10))
    print("Realizations:", phone_counts.most_common(10))
    print("---")

    realizations = [units.index(s) for s,c in phone_counts.most_common(10)]
    prepost_stats = {}
    triphones = {}
    for phone in realizations:
        prepost_stats[phone] = (Counter(), Counter())
        triphones[phone] = Counter()

    for phone, pre, post in preds:
        if phone in prepost_stats.keys():
            prepost_stats[phone][0][pre] += 1
            prepost_stats[phone][1][post] += 1
        if phone in triphones.keys():
            if pre > 0 and post < len(units):
                tri = "".join([units[pre], units[phone], units[post]])
            elif pre > 0:
                tri = "".join([units[pre], units[phone]])
            elif post < len(units):
                tri = "".join([units[phone], units[post]])
            else:
                tri = units[phone]
            triphones[phone][tri] += 1


    for phone in prepost_stats.keys():
        print("realization:", units[phone])
        pre = [(units[p], c) for p,c in prepost_stats[phone][0].most_common() if p < len(units) and p >= 0 and p > 21]
        post = [(units[p], c) for p,c in prepost_stats[phone][1].most_common() if p < len(units) and p >= 0 and p > 21]
        print("pre:", pre[:10])
        print("post:", post[:10])
        print("triphones:", triphones[phone].most_common(10))
        print("---")

if __name__ == "__main__":

    main(args)
