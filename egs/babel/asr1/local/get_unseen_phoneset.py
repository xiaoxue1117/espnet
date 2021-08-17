#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import epitran
import multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser

args = get_parser().parse_args()

def main(args):

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)["utts"]

    n = 158
    phone_mask = [0]*n

    out_of_vocabs = 0
    total = 0

    for u in data:
        tokenid = data[u]["output"][-1]["tokenid"]
        for t in tokenid:
            if t != -1:
                phone_mask[int(t)] = 1
                total += 1
            else:
                out_of_vocabs += 1

    print(phone_mask)
    print(sum(phone_mask))
    for i in range(len(phone_mask)):
        print(i, phone_mask[i])
    print("unks", str(out_of_vocabs))
    print("unks %", str(out_of_vocabs / total))

if __name__ == "__main__":
    main(args)
