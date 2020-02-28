from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

x=np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])

y=np.array([0,0,0,0,0,1,1,1,1,1])

ss=StratifiedShuffleSplit(n_splits=2,test_size=0.25,random_state=42)
for train_index,test_index in ss.split(x,y):
    print("train_index",train_index)
    print("test_index",test_index)



