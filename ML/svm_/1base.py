from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs

# x,y=make_circles(100,factor=0.1,noise=.1)
# plt.scater(x[:,0],x[:,1],c=y,s=50,cmap="rainbowl")


x,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)

print("x",x.shape,y.shape)

plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap="rainbow")#rainbow彩虹色
# plt.xticks([])
# plt.yticks([])
# plt.show()

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()
print("xlim",xlim,ylim)

axisx=np.linspace(xlim[0],xlim[1],30)
axisy=np.linspace(ylim[0],ylim[1],40)


axisy,axisx=np.meshgrid(axisy,axisx)
print("axisx",axisx.shape)
print("axisy",axisy.shape)

xy=np.vstack([axisx.ravel(),axisy.ravel()]).T
print("xy",xy.shape)

# 建模，通过fit计算出对应的决策边界
clf = SVC(kernel="linear").fit(x, y)  # 计算出对应的决策边界
Z = clf.decision_function(xy).reshape(axisx.shape)
# 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
# 然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致
print("Z",axisx.shape,axisy.shape,Z.shape)
# 首先要有散点图
# plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="rainbow")
ax = plt.gca()  # 获取当前的子图，如果不存在，则创建新的子图
# 画决策边界和平行于决策边界的超平面
ax.contour(axisx, axisy, Z
           , colors="k"
           , levels=[-1, 0, 1]  # 画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
           , alpha=0.5  # 透明度
           , linestyles=["--", "-", "--"])

ax.set_xlim(xlim)  # 设置x轴取值
ax.set_ylim(ylim)
print("point",clf.support_vectors_)


plt.show()







