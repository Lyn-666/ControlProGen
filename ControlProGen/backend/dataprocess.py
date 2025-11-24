import pandas as pd
import itertools
from scipy.stats import mode

def df_pair_cvs(df_file):
    src = []
    tgt = []
    label = []
    combination = list(itertools.combinations(range(len(df_file['sequence'])), 2)) # create pairs 

    for i in combination:
        ## creating sequence pair, the input sequence have lower attribution value then target.
        if df_file['sequence'][i[0]] == df_file['sequence'][i[1]]:
            continue

        if df_file['label'][i[0]] <= df_file['label'][i[1]]:
            src.append(df_file['sequence'][i[0]])
            tgt.append(df_file['sequence'][i[1]])
        else:
            src.append(df_file['sequence'][i[1]])
            tgt.append(df_file['sequence'][i[0]])
        
        # the attribution value boundary
        label.append(abs(df_file['label'][i[0]]-df_file['label'][i[1]]))  
        
    df = pd.DataFrame({"src":src, "tgt":tgt, "label":label})
    return df

def top_k_seq(df_file):
    df_sorted = df_file.sort_values(by='label', ascending=False)
    top_k_seq = list(df_sorted.iloc[0:5]['sequence'])
    return top_k_seq

def max_len(df_file):
    return int(len(df_file['sequence'][0]))

def csv_json_ice(df_file):
    df_sorted = df_file.sort_values(by="label", ascending=False)

    if len(df_file) <= 20000:
        final_df = df_file
    elif len(df_file) <= 60000:
        top_20k=df_sorted.head(20000)
        label_cutoff = df_sorted.iloc[19999]["label"]
        df_mid = df_file[df_file['label']< label_cutoff].sample(n=10000, random_state=42)
        final_df = pd.concat([top_20k, df_mid]).drop_duplicates()
    else:
        top_20k=df_sorted.head(40000)
        label_cutoff = df_sorted.iloc[39999]["label"]
        df_mid = df_file[df_file['label']< label_cutoff].sample(n=10000, random_state=42)
        final_df = pd.concat([top_20k, df_mid]).drop_duplicates()
    
    train = []

    for i, row in final_df.iterrows():
        src = row["src"]
        tgt = row["tgt"]
        
        #  skip pairs: src == tgt
        if src == tgt:
            continue

        train.append({
            "src": src,
            "tgt": tgt
        })
    return train   