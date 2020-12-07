#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="modify asr jsons to have two targets")
    parser.add_argument("--input-json", required=True, type=str)
    parser.add_argument("--output-json", required=True, type=str)
    parser.add_argument("--sp", default=False, action='store_true') 
    return parser

args = get_parser().parse_args()

def main(args, start, end):

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for u in data["utts"].keys():
        data["utts"][u]["category"] = u[start:end]

    with open(args.output_json, "wb") as json_file:
        json_file.write(
                json.dumps(
                    data, indent=4, ensure_ascii=False, sort_keys=True
                ).encode("utf_8")
        )

if __name__ == "__main__":
    if args.sp == True:
        start = 6
        end = 9
    else:
        start = 0
        end = 3

    main(args, start, end)
