#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import epitran
import numpy as np
from itertools import groupby


def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--phoneme-units-dir", required=True, type=str)
    parser.add_argument("--phone-units", required=True, type=str)
    parser.add_argument("--bpe-units", required=True, type=str)
    return parser


args = get_parser().parse_args()


def collapse(seq):
    return [k for k, g in groupby(seq)]


def main(args):

    bpes = ["<blank>"]
    with open(args.bpe_units, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for li in lines:
            name, idx = li.split()
            bpes.append(name)
    bpes.append("sos")

    phones = ["<blank>"]
    with open(args.phone_units, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for li in lines:
            try:
                name, idx = li.split()
            except:
                name = ""
                idx = li.split()[0]
            phones.append(name)
    phones.append("sos")

    #langs = ["105", "106", "107", "302", "307", "402", "000"]
    langs = ["105", "106", "107", "302", "307", "402"]
    phonemes = {}
    for lid in langs:
        phonemes[lid] = ["<blank>"]
        with open(
            args.phoneme_units_dir + lid + "_ipa.txt", "r", encoding="utf-8"
        ) as f:
            lines = f.readlines()
            for li in lines:
                name, idx = li.split()
                phonemes[lid].append(name)
        phonemes[lid].append("sos")

    with open(args.input_json, "r", encoding="utf-8") as f, open(
        args.output, "w", encoding="utf-8"
    ) as out_f:
        data = json.load(f)["utts"]
        for utt in data.keys():
            align = data[utt]["output"][-1]
            lid = utt[0:3]
            if lid == "sw0":
                lid = "000"
            n = align["phones"]
            m = align["phonemes"]
            t = align["tokens"]

            # new_t = t
            # last = 0
            # for i in range(len(new_t)-1, -1, -1):
            #    if new_t[i] == 0:
            #        new_t[i] = last
            #    else:
            #        last = new_t[i]

            # start = 0
            # last = new_t[0]
            # toks = []
            # for i in range(1, len(new_t)):
            #    if i == len(new_t) - 1:
            #        toks.append((start, i, last))
            #    elif new_t[i] != last:
            #        toks.append((start, i-1, last))
            #        start = i
            #    last = new_t[i]

            start = 0
            last = t[0]
            mem = last
            inside = False
            toks = []
            for i in range(1, len(t)):
                if i == len(t) - 1:
                    toks.append((start, i, mem))
                elif t[i] == 0:
                    pass
                elif inside:
                    if t[i] != last:
                        toks.append((start, i - 1, mem))
                        start = i
                        mem = t[i]
                else:
                    inside = True
                    mem = t[i]
                last = t[i]

            out_f.write(utt + "\n")
            out_f.write("ref: " + data[utt]["output"][0]["token"] + "\n")
            out_f.write("hyp: " + data[utt]["output"][0]["rec_token"] + "\n")
            ph_gt = data[utt]["output"][1]["token"].split(" ")
            ph_gt = "".join([phonemes[lid][int(x)] for x in ph_gt])
            out_f.write("ref_ph: " + ph_gt + "\n")
            ph_hp = data[utt]["output"][1]["rec_token"].split(" ")
            try:
                ph_hp = "".join([phonemes[lid][int(x)] for x in ph_hp])
            except ValueError:
                ph_hp = ""
            out_f.write("hyp_ph: " + ph_hp + "\n")
            all_ns = ""
            for t in toks:
                n_t = collapse(n[t[0] : t[1] + 1])
                m_t = collapse(m[t[0] : t[1] + 1])
                t_i = t[2]
                ns_list = [phones[x] for x in n_t if x != 0]
                ns = "".join(ns_list)
                all_ns = all_ns + ns
                ms = "".join([phonemes[lid][x] for x in m_t if x != 0])
                ts = bpes[t_i]
                line = "\t||\t".join([ts, ns, ms])

                flag = ""
                if ns != ms:
                    flag = flag + "!"
                for x in ns_list:
                    if x not in phonemes[lid]:
                        flag = flag + "?"
                out_f.write(flag + line + "\n")
            out_f.write("phones: " + all_ns + "\n")

            out_f.write("----------\n")


if __name__ == "__main__":
    main(args)
