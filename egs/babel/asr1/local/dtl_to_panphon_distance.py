#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from panphon import featuretable

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    parser.add_argument("--weighted", type=int, default=0)
    return parser

args = get_parser().parse_args()

ft = featuretable.FeatureTable()

proxy = {'dʒ':'d', 'ɡ̥':'ɡ', 'kx':'k', 'tʃ':'t', 'ă':'a', 'g':'ɡ', 'p̚':'p', 'k͡p̚':'k', 't̪̚':'t', 'ɑ̱':'ɑ', 'gʷ':'ɡ', 'ь':'o', 'b̥':'b', 'ŋ͡m':'ŋ', 'k̟̚':'k', 'ә':'e', 'd̯':'d', 'd̥':'d'}

def dist(a, b):
    seg_a = ft.word_fts(a)[0]
    try:
        seg_b = ft.word_fts(b)[0]
    except:
        import pdb;pdb.set_trace()
    if args.weighted == 1:
        return seg_a.weighted_distance(seg_b)
    return seg_a.distance(seg_b)

dictionary = {}
with open(args.units, encoding="utf-8") as f:
    for l in f:
        try:
            symbol, val = l.strip().split()
        except:
            symbol = ""
            val = l.strip()
        if int(val) > 24:
            dictionary[symbol] = val

if __name__ == "__main__":
    counts = []
    dists = []
    unaccounted = 0
    non_phones = {}
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()[31:]  #double check that this is start of confusions
        for i, li in enumerate(lines):
            if "->" not in li:  #signals end of confusions
                break

            try:
                _, cnt, _, ref, _, hyp = li.strip().split()
            except:
                print(li)
                continue

            #some are not in panphon, so we need proxy
            if ref in proxy.keys():
                ref = proxy[ref]
            if hyp in proxy.keys():
                hyp = proxy[hyp]

            if ref in ft.seg_dict.keys() and hyp in ft.seg_dict.keys():
                d = dist(ref, hyp)
                counts.append(int(cnt))
                dists.append(d)
            else:
                unaccounted += int(cnt)
                if ref not in ft.seg_dict.keys():
                    if ref in non_phones.keys():
                        non_phones[ref] += int(cnt)
                    else:
                        non_phones[ref] = int(cnt)
                if hyp not in ft.seg_dict.keys():
                    if hyp in non_phones.keys():
                        non_phones[hyp] += int(cnt)
                    else:
                        non_phones[hyp] = int(cnt)

        denom = sum(counts)
        num = 0
        for c, d in zip(counts, dists):
            num += c * d
        print("confusion distance", str(num / denom))
        print("unaccounted", str(unaccounted / denom))
        print(non_phones)
        print(dists)
