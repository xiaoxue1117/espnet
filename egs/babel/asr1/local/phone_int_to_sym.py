#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import epitran
import multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    return parser

dictionary={}
args = get_parser().parse_args()

def main(target):
    with open(args.dir+"/"+target+".trn", "r", encoding="utf-8") as f1, open(args.dir+"/"+target+".trn.sym", "w", encoding="utf-8") as f2:
        lines = f1.readlines()
        for li in lines:
            try:
                toks, uttid = li.strip().rsplit(' ', 1)
            except:
                print(li)
                f2.write(li)
                continue
            syms = []
            for t in toks.split():
                if int(t) == -1:
                    syms.append('unk')
                else:
                    try:
                        syms.append(dictionary[t])
                    except:
                        import pdb;pdb.set_trace()
            syms_txt = " ".join(syms)
            f2.write(syms_txt + ' ' + uttid + '\n')

if __name__ == "__main__":
    # Get dictionary 
    with open(args.units, encoding="utf-8") as f:
        for l in f:
            try:
                symbol, val = l.strip().split()
            except:
                symbol = ""
                val = l.strip()
            dictionary[val] = symbol

    main('hyp')
    main('ref')
