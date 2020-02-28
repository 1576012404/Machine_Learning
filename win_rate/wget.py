
import os
import os.path
import pandas as pd
import urllib.request
import multiprocessing
import json

df_replay = pd.read_csv(os.path.join("data", "replay_list.csv"), index_col=0)
# step3 下载replay文件

def worker(worker_index,iStart,iEnd,iMaxMayDown):
    file_name = 'data/down_worker_%s.json' % worker_index
    if os.path.exists(file_name):
        print("exist")
        with open(file_name, 'r') as load_f:
            load_dict = json.load(load_f)
            iStart=load_dict["To"]

    for i in range(iStart,iEnd):
        match_id = df_replay["match_id"][i]
        cluster = df_replay["cluster"][i]
        replay_salt = df_replay["replay_salt"][i]
        url = "http://replay%s.valve.net/570/%s_%s.dem.bz2"%(cluster, match_id, replay_salt)
        dest = "./replays/%s_%s.dem.bz2"%(match_id, replay_salt) # or ‘~/Downloads/’ on linux
        print("start",i, url)

        if i<iMaxMayDown:
            if not os.path.exists(dest):
                try:
                    urllib.request.urlretrieve(url, dest)
                    print("D", i, url, dest)
                except urllib.error.URLError as e:
                    print(e.reason, i, url)
        else:
            try:
                urllib.request.urlretrieve(url, dest)
                print("D", i, url, dest)
            except urllib.error.URLError as e:
                print(e.reason, i, url)




        if i%20==0:
            data = {"To":i}
            datas = json.dumps(data)  # ensure_ascii：使用中文保存，缩进为4个空格
            with open(file_name, 'w') as f:
                f.write(datas)

    #wget.download(url, dest)



if __name__=="__main__":
    iMax=df_replay.shape[0]
    iProcess=1
    Gap=iMax//iProcess
    jobs=[]
    worker(0,0,5000,0)

    # for i in range(iProcess):
    #     iStart=i*Gap
    #     iEnd=(i+1)*Gap
    #     p = multiprocessing.Process(target=worker, args=(i,iStart,iEnd))
    #     jobs.append(p)
    #     p.start()
