import epitran
import sys
import multiprocessing as mp

epitran_code_dict = {
    "it": "ita-Latn",
    "de": "deu-Latn",
    "fr": "fra-Latn",
    "nl": "nld-Latn",
    "es": "spa-Latn",
    "105": "tur-Latn",
    "106": "tgl-Latn",
    "107": "vie-Latn",
}

epi = epitran.Epitran(epitran_code_dict[sys.argv[1]])

if sys.argv[2] == "punct":
    epi_fn = epi.trans_list
elif sys.argv[2] == "char":
    epi_fn = "char"
else:
    epi_fn = epi.trans_list

non_langs = []
if len(sys.argv) > 3:
    non_lang_sym_file = open(sys.argv[3]).readlines()
    for l in non_lang_sym_file:
        non_langs.append(l.strip())


def get_epitran(line):
    # return " ".join(epi.trans_list(line))
    # if sys.argv[3] == "space":
    #    words = line.split(" ")
    # else:
    #    words = line.split(" ")
    #    return " ".join(epi_fn(w)) for w in words])
    if epi_fn == "char":
        # words = [c for w in line.split(" ") if w not in non_langs for c in w]
        words = []
        for w in line.split(" "):
            if w not in non_langs:
                temp = [c for c in w]
                words.append(" ".join(temp))
    else:
        words = [" ".join(epi_fn(w)) for w in line.split(" ") if w not in non_langs]
    return " ".join([w for w in words if w != ""])


lines = []
prev_line = ""
for line in sys.stdin:
    if line == prev_line:
        continue
    lines.append(line)

with mp.Pool(processes=20) as p:
    results = p.map(get_epitran, lines)

for result in results:
    print(result)
