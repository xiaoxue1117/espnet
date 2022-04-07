import pandas as pd
import numpy as np

df = pd.read_csv('downloads/babel_107/conversational/reference_materials/demographics.tsv', sep='\t')

dico = {}
for i in range(len(df)):
    name = df['outputFn'][i]
    l=name.split('_')
    if "inLine.sph" in name:
        letter="A"
    else :
        letter="B"

    gender = df["gen"][i]

    s = "107_{}_{}_{}_{}".format(l[3],letter,l[4],l[5])
    dico[s]=gender



for spl in range(1,2):
    nf = "exp_semi_supervised_gender_10h/asr_stats_raw_107_bpe1000/splits2/text/split.{}".format(spl)

    with open(nf,"r") as f:
        text = f.readlines()

    text2=""
    for line in text:
        uttid = line.split()[0]
        t = line.split()[1]

        first = "_".join(uttid.split('_')[:-1])
        if "sp" in first:
            first=first.split("-")[1]
        gender = dico[first]

        text2+=uttid + " " + gender + "\n"

    with open(nf,"w") as f:
        f.write(text2)