import pandas as pd
import json
# elif x["type"]=="epilogue":
#    s=json.loads(x["key"])
#    winner=s["gameInfo_"]["dota_"]["gameWinner_"]
#    playbackFrames=s["playbackFrames_"]
#    playbackTime=s["playbackTime_"]

save_dict={}
with open("items.json","r") as f:
    load_dict = json.load(f)
    print("load_dict",load_dict)

    item_list=load_dict["result"]["items"]
    print("iter",item_list)
    for id_dct in item_list:
        name=id_dct["name"]
        iId=id_dct["id"]

        save_dict[name]=iId
    print("save_dict",len(save_dict),save_dict)

with open("item.json", 'w') as f:
    datas = json.dumps(save_dict)
    f.write(datas)


from collections import defaultdict


# a=defaultdict(lambda :defaultdict(int))
# print(a)
#
# a[1][3]+=3
# a[2][4]+=1
# print("a",a)

