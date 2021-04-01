#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import epitran
import multiprocessing as mp

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--units", required=True, type=str)
    parser.add_argument("--text-file", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    return parser

dictionary={}
args = get_parser().parse_args()

def main(args, token_dict, token_id_dict, shape_dict):

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for u in token_dict.keys():
        temp_dict = copy.copy(data["utts"][u]["output"][0])
        temp_dict["name"] = "target2"
        temp_dict["token"] = token_dict[u]
        temp_dict["tokenid"] = token_id_dict[u]
        temp_dict["shape"] = shape_dict[u]
        data["utts"][u]["output"].append(temp_dict)
        data["utts"][u]["category"] = "000"

    with open(args.output_json, "wb") as json_file:
        json_file.write(
                json.dumps(
                    data, indent=4, ensure_ascii=False, sort_keys=True
                ).encode("utf_8")
        )

problems = {}

def get_ph_dets(l):
    if len(l.strip().split(" ")) == 1:
        utt = l.strip().split(" ",1)[0]
        text= ""
    else:
        utt, text = l.strip().split(" ",1)

    tokenids = []
    # this is hacked: tusom only has phones that are 1 char.
    if args.lang == "tus":
        for w in text.split(" "):
            for phone in w:
                if phone in dictionary.keys():
                    tokenids.append(dictionary[phone])
                else:
                    tokenids.append(-1)
    # inuktitut data is a series of phones, space separated
    elif args.lang == "inu":
        for w in text.split(" "):
            if w == "<sp>":
                tokenids.append(dictionary[w])
            else:
                tmp = " ".join([c for c in w])
                for problem in problems.keys():
                    tmp_fixed = tmp.replace(problem, problems[problem])
                phones = tmp_fixed.split(" ")
                for phone in phones:
                    if phone in dictionary.keys():
                        tokenids.append(dictionary[phone])
                    else:
                        tokenids.append(-1)

    token = text
    token_id = tokenids
    shape = [len(tokenids) , len(dictionary)+2]
    return (utt,token,token_id,shape)


if __name__ == "__main__":
    # Get dictionary 
    with open(args.units, encoding="utf-8") as f:
        for l in f:
            try:
                symbol, val = l.strip().split(" ")
            except:
                symbol = ""
                val = l.strip()
            dictionary[symbol] = val
    shape_dict = {}
    token_dict = {}
    token_id_dict = {}
    text_lines = []
    with open(args.text_file,encoding="utf-8") as f:
        for l in f:
            text_lines.append(l)

    with open("/project/ocean/xinjianl/corpus/iku/inuktitut_agg/langs/ipa/units.txt", encoding="utf-8") as f:
        for l in f:
            unit, _ = l.split()
            if len(unit) == 2:
                unitsp = unit[0]+" "+unit[1]
                problems[unitsp] = unit

    with mp.Pool(processes = 20) as p:
        results = p.map(get_ph_dets, text_lines)
    #for l in text_lines:
    #    get_ph_dets(l)

    for result in results:
        utt = result[0]
        token_dict[utt] = result[1]
        token_id_dict[utt] = result[2]
        shape_dict[utt] = result[3]

    main(args,token_dict, token_id_dict, shape_dict)
