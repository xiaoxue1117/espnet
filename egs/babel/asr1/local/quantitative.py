#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from jiwer import wer

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--ref", required=True, type=str)
    parser.add_argument("--hyp1", required=True, type=str)
    parser.add_argument("--hyp2", required=True, type=str)
    parser.add_argument("--hyp3", required=True, type=str)
    return parser

args = get_parser().parse_args()

import numpy as np
import editdistance

class Utils:
    @staticmethod
    def normalize(d, total):
        """
        Converting raw counts to frequencies
        :param d: collection of counts to be normalized
        :param total: normalization constant
        :return: normalized version of the input
        """

        obj_type = type(d)
        dn = obj_type()
        for key in d.keys():
            dn[key] = float(d[key]) / total
        return dn

    @staticmethod
    def editops(src_list, trg_list):

        chart = np.zeros((len(src_list) + 1, len(trg_list) + 1))
        bp = np.zeros((len(src_list) + 1, len(trg_list) + 1))
        # ins = 1, sub = 2, del = 3
        # down = delete, right = insert

        for j in range(1, len(trg_list) + 1):
            chart[0][j] = j
            bp[0][j] = 1

        for i in range(1, len(src_list) + 1):
            chart[i][0] = i
            bp[i][0] = 3

        for i in range(1, len(src_list) + 1):
            for j in range(1, len(trg_list) + 1):
                if src_list[i - 1] == trg_list[j - 1]:
                    chart[i][j] = chart[i - 1][j - 1]
                    bp[i][j] = 0
                else:
                    chart[i][j] = chart[i - 1][j - 1] + 1
                    bp[i][j] = 2
                    if chart[i][j - 1] + 1 < chart[i][j]:
                        chart[i][j] = chart[i][j - 1] + 1
                        bp[i][j] = 1
                    if chart[i - 1][j] + 1 < chart[i][j]:
                        chart[i][j] = chart[i - 1][j] + 1
                        bp[i][j] = 3

        i = len(src_list)
        j = len(trg_list)
        ops = []

        # assert chart[i][j] == editdistance.eval(src_list, trg_list)
        while i > 0 or j > 0:
            if bp[i, j] == 1:
                ops.append(("insert", "INS->{}".format(trg_list[j - 1])))
                j -= 1
            elif bp[i, j] == 2:
                ops.append(("replace", "{}->{}".format(src_list[i - 1], trg_list[j - 1])))
                j -= 1
                i -= 1
            elif bp[i, j] == 3:
                ops.append(("delete", "{}->DEL".format(src_list[i - 1])))
                i -= 1
            else:
                j -= 1
                i -= 1

        return ops[::-1]

    def get_rate(edit_type, src_list, trg_list):
        c = 0
        errs = Utils.editops(src_list, trg_list)
        errs_in_type = []
        for o in errs:
            if o[0] == edit_type:
                c += 1
                errs_in_type.append(o)
        return c / len(src_list), errs_in_type
        #return c / len(src_list), errs

from panphon import featuretable
ft = featuretable.FeatureTable()

proxy = {'dʒ':'d', 'ɡ̥':'ɡ', 'kx':'k', 'tʃ':'t', 'ă':'a', 'g':'ɡ', 'p̚':'p', 'k͡p̚':'k', 't̪̚':'t', 'ɑ̱':'ɑ', 'gʷ':'ɡ', 'ь':'o', 'b̥':'b', 'ŋ͡m':'ŋ', 'k̟̚':'k', 'ә':'e', 'd̯':'d', 'd̥':'d'}

def dist(a, b):
    try:
        seg_a = ft.word_fts(a)[0]
        seg_b = ft.word_fts(b)[0]
    except:
        return 0
    return seg_a.distance(seg_b)

def get_afd(subs):
    if len(subs) == 0:
        return 0
    num = 0
    denom = 0
    for err_type, sub in subs:
        if err_type != 'replace':
            continue
        a, b = sub.split("->")
        if a in proxy:
            a = proxy[a]
        if b in proxy:
            b = proxy[b]
        num += dist(a, b)
        denom += 1
    if denom == 0:
        return 0
    return num / denom

if __name__ == "__main__":
    with open(args.ref, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    with open(args.hyp1, "r", encoding="utf-8") as f:
        hyp1_lines = f.readlines()
    with open(args.hyp2, "r", encoding="utf-8") as f:
        hyp2_lines = f.readlines()
    with open(args.hyp3, "r", encoding="utf-8") as f:
        hyp3_lines = f.readlines()

    for r, h1, h2, h3 in zip(ref_lines, hyp1_lines, hyp2_lines, hyp3_lines):
        try:
            r, uttid = r.strip().rsplit(' ', 1)
            h1, _ = h1.strip().rsplit(' ', 1)
            h2, _ = h2.strip().rsplit(' ', 1)
            h3, _ = h3.strip().rsplit(' ', 1)
        except:
            continue
        cer1 = wer(r, h1)   #hyp and ref are split by char, so just call wer fxn to get the cer
        cer2 = wer(r, h2)
        cer3 = wer(r, h3)
        # print examples where h1 (AG+UC) < h2 (AG) < h3 (AM)
        if cer1 < cer2 and cer2 < cer3:
            ser1, subs1 = Utils.get_rate("replace", r.split(), h1.split())
            ser2, subs2 = Utils.get_rate("replace", r.split(), h2.split())
            ser3, subs3 = Utils.get_rate("replace", r.split(), h3.split())

            afd1 = get_afd(subs1)
            afd2 = get_afd(subs2)
            afd3 = get_afd(subs3)

            print(uttid)
            print("".join(r.split()))
            print("".join(h1.split())+"\t"+str(cer1)[:5]+"\t"+str(ser1)[:5]+"\t"+str(afd1)[:5]+"\t"+str(subs1))
            print("".join(h2.split())+"\t"+str(cer2)[:5]+"\t"+str(ser2)[:5]+"\t"+str(afd2)[:5]+"\t"+str(subs2))
            print("".join(h3.split())+"\t"+str(cer3)[:5]+"\t"+str(ser3)[:5]+"\t"+str(afd3)[:5]+"\t"+str(subs3))
