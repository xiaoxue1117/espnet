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

with open("dump/dump_lid_107/raw/train_107_10h/text","r") as f:
    text = f.readlines()

DICO_UTT={}
text2=""
utt2cat=""
for line in text:
    uttid = line.split()[0]
    t = " ".join(line.split()[1:])

    first = "_".join(uttid.split('_')[:-1])
    gender = dico[first]
    
    if np.random.random()>0.5:
        DICO_UTT[uttid]=uttid+"_SEMI_{}".format(gender)
        uttid+="_SEMI_{}".format(gender)
        utt2cat+=uttid + " " + "UNLABEL" + "\n"

    else : 
        utt2cat+=uttid + " " + "LABEL" + "\n"
        DICO_UTT[uttid]=uttid
    text2+=uttid + " " + t + "\n"

with open("dump/dump_lid_107/raw/train_107_10h_semi_gender/text","w") as f:
    f.write(text2)
with open("dump/dump_lid_107/raw/train_107_10h_semi_gender/utt2category","w") as f:
    f.write(utt2cat)


for fichier in ["wav.scp", "utt2dur", "utt2spk","utt2num_samples"]:
    with open("dump/dump_lid_107/raw/train_107_10h/{}".format(fichier),"r") as f:
        wav = f.readlines()
    wav2=""
    for line in wav:
        a,b = line.split()
        wav2 += "{} {}\n".format(DICO_UTT[a],b)
    with open("dump/dump_lid_107/raw/train_107_10h_semi_gender/{}".format(fichier),"w") as f:
        f.write(wav2)

# then you need to generate spk2utt from utt2spk 