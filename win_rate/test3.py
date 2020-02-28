import numpy as np

a=np.arange(12).reshape(3,4)
b=np.array([0,1,0])

c=np.array([2,3,4])

d=np.concatenate((b,c))
print(d.shape,d)