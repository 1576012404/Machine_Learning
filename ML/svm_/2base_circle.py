from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.svm import  SVC
import numpy as np

def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    x=np.linspace(xlim[0],xlim[1],30)
    y=np.linspace(ylim[0],ylim[1],30)
    y,x=np.meshgrid(y,x)
    xy=np.vstack([x.ravel(),y.ravel()]).T
    p=model.decision_function(xy).reshape(x.shape)
    ax.contour(x,y,p,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)





x,y=make_circles(100,factor=0.1,noise=.1)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap="rainbow")

clf=SVC(kernel="linear").fit(x,y)

plot_svc_decision_function(clf)
score=clf.score(x,y)
print("score",score)

r=np.exp(-x**2).sum(1)
print("r",x.shape,r.shape)

rlim=np.linspace(min(r),max(r),100)
print("rlim",rlim.shape)



from mpl_toolkits import mplot3d

def plot_3D(elev=30,azim=30,x=x,y=y):
    ax=plt.subplot(projection="3d")
    ax.scatter3D(x[:,0],x[:,1],r,c=y,s=50,cmap="rainbow")
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()



plot_3D()


# plt.show()


