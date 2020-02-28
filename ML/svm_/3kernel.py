import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_circles,make_moons,make_blobs,make_classification

n_samples=100
datasets=[
    make_moons(n_samples=n_samples,noise=0.2,random_state=0),
    make_circles(n_samples=n_samples,noise=0.2,factor=0.5,random_state=1),
    make_blobs(n_samples=n_samples,centers=2,random_state=5),
    make_classification(n_samples=n_samples,n_features=2,n_informative=2,n_redundant=0,random_state=5)
]
Kernel=["linear","poly","rbf","sigmoid"]
# for x,y in datasets:
#     plt.figure(figsize=(5,4))
#     plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap="rainbow")
#
# plt.show()

nrows=len(datasets)
ncols=len(Kernel)+1

fig,axes=plt.subplots(nrows,ncols,figsize=(20,16))


for ds_cnt,(x,y) in enumerate(datasets):
    ax=axes[ds_cnt,0]
    if ds_cnt==0:
        ax.set_title("input data")
    ax.scatter(x[:,0],x[:,1],c=y,zorder=10,cmap=plt.cm.Paired,edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())

    for est_idx,kernel in enumerate(Kernel):
        ax=axes[ds_cnt,est_idx+1]
        clf=SVC(kernel=kernel,gamma=2).fit(x,y)
        score=clf.score(x,y)
        ax.scatter(x[:,0],x[:,1],c=y,zorder=10,cmap=plt.cm.Paired,edgecolors="k")
        ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=50,
                   facecolors="none",zorder=10,edgecolors="k")
        x_min,x_max=x[:,0].min()-0.5,x[:,0].max()+0.5
        y_min,y_max=x[:,1].min()-0.5,x[:,1].max()+0.5

        xx,yy=np.mgrid[x_min:x_max:200j,y_min:y_max:200j]
        z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)

        ax.pcolormesh(xx,yy,z>0,cmap=plt.cm.Paired)
        ax.contour(xx,yy,z,colors=["k","k","k"],linestyles=["--","-","--"],
                   levels=[-1,0,1])
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt==0:
            ax.set_title(kernel)
        ax.text(0.95,0.06,("%.2f"%score).lstrip("0"),
                size=15,
                bbox=dict(boxstyle="round",alpha=0.8,facecolor="white"),
                transform=ax.transAxes,
                horizontalalignment="right"
                )
plt.tight_layout()
plt.show()

