import os
import multiprocessing
import glob

def worker(name_list):
    for filename in name_list:
        dirname = "@/home/liu/dota/%s.dem" % filename
        os.system("curl localhost:5600 --data-binary %s > x.json" % dirname)

if __name__=="__main__":
    print("enter")
    jobs=[]
    name_list_all=glob.glob("*.dem")
    iProcessNum=8
    iGap=len(name_list_all)//iProcessNum

    for i in range(iProcessNum+1):
        if i<iProcessNum:
            name_list=name_list_all[i*iGap:(i+1)*iGap]
        else:
            name_list=name_list_all[i*iGap:]
        p = multiprocessing.Process(target=worker, args=(i,name_list))
        jobs.append(p)
        p.start()