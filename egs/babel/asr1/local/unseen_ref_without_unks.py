#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import epitran
import multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    parser.add_argument("--text-file", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    return parser

dictionary={}
tus_problems={}
inu_problems={}
args = get_parser().parse_args()

def get_line(l):
    if len(l.strip().split(" ")) == 1:
        utt = l.strip().split(" ",1)[0]
        text= ""
    else:
        utt, text = l.strip().split(" ",1)

    tokenids = []
    # this is hacked: tusom only has phones that are 1 char.
    if args.lang == "tus":
        for w in text.split(" "):
            tmp = " ".join([c for c in w])
            tmp_fixed = tmp
            for problem in tus_problems.keys():
                tmp_fixed = tmp_fixed.replace(problem, tus_problems[problem])
            phones = tmp_fixed.split(" ")
            for phone in phones:
                tokenids.append(phone)
        #fix uttid
        prefix = utt.split('_')[-1].replace('-','_')
        utt=prefix+'-'+utt

    # inuktitut data is a series of phones, space separated
    elif args.lang == "inu":
        for w in text.split(" "):
            if w == "<sp>":
                continue
                #tokenids.append(dictionary[w])
            else:
                tmp = " ".join([c for c in w])
                tmp_fixed = tmp
                for problem in inu_problems.keys():
                    tmp_fixed = tmp_fixed.replace(problem, inu_problems[problem])
                phones = tmp_fixed.split(" ")
                for phone in phones:
                    tokenids.append(phone)

        #fix uttid
        prefix = utt.split('_')[-1]
        utt=prefix+'-'+utt

    return ' '.join(tokenids) + ' (' + utt + ')\n'

if __name__ == "__main__":
    # Get dictionary 
    with open(args.units, encoding="utf-8") as f:
        for l in f:
            try:
                symbol, val = l.strip().split()
            except:
                symbol = ""
                val = l.strip()
            dictionary[symbol] = val

    text_lines = []
    with open(args.text_file,encoding="utf-8") as f:
        for l in f:
            text_lines.append(l)

    with open("/project/ocean/xinjianl/corpus/tusom/tusom_ipa/langs/ipa/units.txt", encoding="utf-8") as f:
        for l in f:
            unit, _ = l.split()
            if len(unit) == 2:
                unitsp = unit[0]+" "+unit[1]
                tus_problems[unitsp] = unit
            elif len(unit) == 2:
                unitsp = unit[0]+" "+unit[1]+" "+unit[2]
                tus_problems[unitsp] = unit

    with open("/project/ocean/xinjianl/corpus/iku/inuktitut_agg/langs/ipa/units.txt", encoding="utf-8") as f:
        for l in f:
            unit, _ = l.split()
            if len(unit) == 2:
                unitsp = unit[0]+" "+unit[1]
                inu_problems[unitsp] = unit

    with open(args.output, 'w', encoding='utf-8') as f:
       for l in text_lines:
           f.write(get_line(l))
