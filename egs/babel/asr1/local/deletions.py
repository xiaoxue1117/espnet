import json
import sys
from analysis_utils import Utils as auutils

fname = sys.argv[1]

with open(fname, "r", encoding="utf-8") as f:
    data = json.load(f)["utts"]

del_dict = {"105": {}, "106": {}, "107": {}}

for utt in data.keys():
    cat = utt[:3]
    hyp = data[utt]["output"][1]["rec_token"]
    ref = data[utt]["output"][1]["token"]
    rate, examples = auutils.get_del_list(ref.split(), hyp.split())
    for e in examples:
        if e in del_dict[cat].keys():
            del_dict[cat][e] = del_dict[cat][e] + 1
        else:
            del_dict[cat][e] = 1

with open(sys.argv[2], "wb") as json_file:
    json_file.write(
        json.dumps(del_dict, indent=4, ensure_ascii=False, sort_keys=True).encode(
            "utf_8"
        )
    )
