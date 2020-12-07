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
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--text-file", required=True, type=str)
    parser.add_argument("--space", default=False, action='store_true')
    parser.add_argument("--char", default=False, action='store_true')
    parser.add_argument("--no-punct", default=False, action='store_true')
    parser.add_argument("--non-lang-file", default=None , type=str)
    return parser

epitran_code_dict = {'it':'ita-Latn' , 'de':'deu-Latn', 'fr':'fra-Latn', 'nl':'nld-Latn', 'es' : 'spa-Latn'}
dictionary={}
args = get_parser().parse_args()

non_langs=[]
if args.non_lang_file is not None:
    non_lang_sym_file = open(args.non_lang_file).readlines()
    for l in non_lang_sym_file:
        non_langs.append(l.strip())

def main(args, token_dict, token_id_dict, shape_dict):

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for u in data["utts"].keys():
        temp_dict = copy.copy(data["utts"][u]["output"][0])
        temp_dict["name"] = "target2"
        temp_dict["token"] = token_dict[u]
        temp_dict["tokenid"] = token_id_dict[u]
        temp_dict["shape"] = shape_dict[u]
        data["utts"][u]["output"].append(temp_dict)

    with open(args.output_json, "wb") as json_file:
        json_file.write(
                json.dumps(
                    data, indent=4, ensure_ascii=False, sort_keys=True
                ).encode("utf_8")
        )


def get_ph_dets(l):
    if len(l.strip().split(" ")) == 1:
        utt = l.strip().split(" ",1)[0]
        text= ""
    else:
        utt, text = l.strip().split(" ",1)

    if args.char:
        tokens = []
        for w in text.split(" "):
            flag=False
            if w in non_langs:
                tokens.append(w)
                flag=True
            else:
                for c in w:
                    if c != "":
                        tokens.append(c)
                        flag=True
            if flag:
                tokens.append("<sp>")
        tokens = tokens[:-1]
    else:
        if args.no_punct:
            epi_fn = epi.xsampa_list
        else:
            epi_fn = epi.trans_list

        if args.space:
            tokens = []
            for w in text.split(" "):
                flag=False
                if w in non_langs:
                    tokens.append(w)
                    flag=True
                else:
                    for z in epi_fn(w):
                        if z != "":
                            tokens.append(z)
                            flag=True
                if flag:
                    tokens.append("<sp>")
            tokens = tokens[:-1]
            #tokens = [z for w in text.split(" ") for z in epi_fn(w) + ['<sp>'] if z != ""]
        else:
            tokens = []
            for w in text.split(" "):
                if w in non_langs:
                    tokens.append(w)
                else:
                    for z in epi_fn(w):
                        if z != "":
                            tokens.append(z)

    tokenids = [dictionary[t] if (t in dictionary) \
               else dictionary['<unk>'] for t in tokens]

    token = "".join(tokens)
    token_id = ' '.join(tokenids)
    shape = [len(tokenids) , len(dictionary)+2]
    return (utt,token,token_id,shape)


if __name__ == "__main__":
    epi = epitran.Epitran(epitran_code_dict[args.lang])
    # Get dictionary 
    with open(args.units, encoding="utf-8") as f:
        for l in f:
            symbol, val = l.strip().split(" ")
            dictionary[symbol] = val
    shape_dict = {}
    token_dict = {}
    token_id_dict = {}
    text_lines = []
    with open(args.text_file,encoding="utf-8") as f:
        for l in f:
            text_lines.append(l)

    with mp.Pool(processes = 20) as p:
        results = p.map(get_ph_dets, text_lines)

    for result in results:
        utt = result[0]
        token_dict[utt] = result[1]
        token_id_dict[utt] = result[2]
        shape_dict[utt] = result[3]

    main(args,token_dict, token_id_dict, shape_dict)
