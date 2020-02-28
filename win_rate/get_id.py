# coding:utf-8
import requests
import json
import os
import pandas as pd
import tqdm
import time

# key = "DDD512766621099248DE19F1084DD275"
data_dir = "./data"


def get_page_source(url):
    headers = {'Accept': '*/*',
               'Accept-Language': 'en-US,en;q=0.8',
               'Cache-Control': 'max-age=0',
               'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
               'Connection': 'keep-alive',
               'Referer': 'http://www.baidu.com/'
               }
    for x in range(3):
        req = requests.get(url, headers=headers, timeout=60)
        if req and req.status_code == 200:
            print("content",req.content)
            return req.content
        time.sleep((1 + x) * 3)

    return None


# # step1 遍历联赛获取比赛id
# ```
# SELECT
# matches.match_id,
# matches.start_time,
# leagues.leagueid,
# leagues.name leaguename
# FROM matches
# JOIN leagues using(leagueid)
# WHERE TRUE
# AND matches.match_id <= 5248926591
# ORDER BY matches.match_id desc NULLS LAST
# LIMIT 100000
# ```
#
df_matches = pd.read_csv(os.path.join(data_dir, "data_5248926591_8w.csv"))


# step2 获取replay salt
def main():
    df_replay = pd.read_csv(os.path.join(data_dir, "replay_list.csv"), index_col=0)
    # print()
    replay_list = eval(df_replay.to_json(orient='records'))
    for i in range(df_replay.shape[0]+1000, 20000):
        print("i",i)
        request_url = "https://api.opendota.com/api/replays?match_id=" + str(df_matches['match_id'][i])
        print(i, request_url)
        content = get_page_source(request_url).decode("utf-8")
        if content != None:
            replay_list += json.loads(content)
        else:
            print('get error', i, request_url)
            raise
        time.sleep(1)
        if len(replay_list) % 10 == 0:
            df_replay = pd.DataFrame(replay_list)
            df_replay.to_csv(os.path.join(data_dir, "replay_list.csv"))


if __name__ == '__main__':
    while 1:
        try:
            main()
        except:
            print("===================== error =====================")