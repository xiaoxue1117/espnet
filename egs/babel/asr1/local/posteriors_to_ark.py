#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os.path
import kaldi_io
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description="numpy to ark")
    parser.add_argument("--mat-dir", required=True, type=str)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--ark-out", required=True, type=str)
    return parser

args = get_parser().parse_args()

def main(args):
    utts = []
    with open(args.text, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for l in lines:
            try:
                uttid, txt = l.split(" ", 1)
            except:
                uttid = l.strip()
                txt = ""
            utts.append(uttid)

    with kaldi_io.open_or_fd(args.ark_out, "wb") as f:
        for utt in utts:
            if os.path.isfile(args.mat_dir+"/"+utt+".npy"):
                mat = np.load(args.mat_dir+"/"+utt+".npy")
                mat = mat.squeeze(axis=0)
                kaldi_io.write_mat(f, mat, key=utt)
            else:
                import pdb; pdb.set_trace()

if __name__ == "__main__":
    main(args)
