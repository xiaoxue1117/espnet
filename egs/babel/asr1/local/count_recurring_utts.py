#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import Counter

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", required=True, type=str)
    return parser

args = get_parser().parse_args()

def main(args):

    utt_counts = Counter()
    utt_accents = {}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

        for line in lines:
            #uttid, utttext = line.split('\t')
            uttid, utttext = line.split(' ', maxsplit=1)
            uttaccent = uttid[:7]
            utt_counts[utttext] += 1
            if utttext in utt_accents.keys():
                utt_accents[utttext][uttaccent] += 1
            else:
                utt_accents[utttext] = Counter()
                utt_accents[utttext][uttaccent] += 1

    common = utt_counts.most_common(20)
    print(common)
    for utt, count in common:
        print(utt, utt_accents[utt])

if __name__ == "__main__":

    main(args)
