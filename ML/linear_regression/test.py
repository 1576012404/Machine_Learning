import numpy as np
from scipy import linalg


xtrain=np.arange(6).reshape(3,2)
ytrain=np.array([1,0,1])
coef_, _residues, rank_, singular_=linalg.lstsq(xtrain, ytrain)
print("co",coef_)
print("_residues",_residues)
print("rank_",rank_)
print("singular_",singular_)