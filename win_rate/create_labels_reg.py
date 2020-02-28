import pandas as pd
from pathlib import Path
import functools
import joblib

from tqdm import tqdm
source_dir = "train_data/csv"
iIndex = 0

win_df=pd.read_csv("train_data/target_list.csv", index_col=0)

all_tables = []
root = Path(source_dir)

def get_label(bWin,row_num,row):
    iIndex = row["turn"]
    fWinRate=0.5+0.5*iIndex/(row_num-1)
    out=fWinRate if bWin else (1-fWinRate)
    print("out",out,iIndex,row_num)
    return out

def process(file_name):
    sName = file_name.name
    file_path = str(file_name)
    df = pd.read_csv(file_path, index_col=0)
    row_num = len(df)
    df = df.iloc[row_num // 2:]
    row_num = len(df)
    df.reset_index(drop=True,inplace=True)
    df['turn'] = df.index
    print("index",df.index)
    match_id = int(sName.split("_")[0])
    if row_num>1:
        iWin = win_df.loc[match_id, "radiant_win"]
        df["label"] = df.apply(functools.partial(get_label, iWin,row_num), axis=1)
        df.drop(axis=1,columns=["turn"],inplace=True)
        return df
    else:
        print("match_id", match_id, row_num)
        return pd.DataFrame()

with joblib.Parallel(n_jobs=-1) as parallel:
    outpout = parallel(joblib.delayed(process)(file_name)
                       for file_name in tqdm(list(root.glob("*.csv"))))

all_tables=list(filter(lambda i:len(i)>0, outpout))

    # if iIndex==1:
    #     break

all_df = pd.concat(all_tables, axis=0)
all_df.reset_index(drop=True,inplace=True)
all_df.to_csv("train_reg.csv")