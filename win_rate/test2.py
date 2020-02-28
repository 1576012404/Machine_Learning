import pandas as pd
import json
# elif x["type"]=="epilogue":
#    s=json.loads(x["key"])
#    winner=s["gameInfo_"]["dota_"]["gameWinner_"]
#    playbackFrames=s["playbackFrames_"]
#    playbackTime=s["playbackTime_"]

save_dict={}
with open("heros.json","r") as f:
    load_dict = json.load(f)
    print("load_dict",load_dict)

    item_list=load_dict["result"]["heroes"]
    print("hero",len(item_list),item_list)
    for id_dct in item_list:
        name=id_dct["name"]
        iId=id_dct["id"]
        name_list = name.split("_")
        new_name = "_".join(name_list[:4]) + "".join(name_list[4:])
        save_dict[name]=iId
        save_dict[new_name] = iId
    print("save_dict",len(save_dict),save_dict)

with open("hero.json", 'w') as f:
    datas = json.dumps(save_dict)
    f.write(datas)
