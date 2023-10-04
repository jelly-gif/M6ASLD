from sklearn import decomposition,manifold
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import plotly.express as px
except:
    px=None


def scatter(data, dimension="2D", point_size=3, sty='default',
            label=None, title=None, alpha=None, aes_label=None,
            **kwargs):
    """
    This function is to plot scatter plot of embedding points
       
    
    Parameters
    ----------
    data : numpy.array
        A numpy array which has 2 or 3 columns, every row represent a point.
    dimension : str, optional
        Specifiy the dimension of the plot, either "2D" or "3D".
        The default is "2D".
    point_size : float, optional
        Set the size of the points in scatter plot.
        The default is 3.
    sty : str, optional
        Styles of Matplotlib. The default is 'default'.
    label : list or None, optional
        Specifiy the label of each point. The default is None.
    title : str, optional
        Title of the plot. The default is None.
    alpha : float, optional
        The alpha blending value. The default is None.
    aes_label : list, optional
        Set the label of every axis. The default is None.
    **kwargs :
        Other arguments passed to the `matplotlib.pyplot.legend`,
        controlling the plot's legend.

    Returns
    -------
        Scatter plot of either 2D or 3D.

    """

    # Error messages.
    if dimension not in ["2D", "3D"]:
        raise ValueError('Dimension must be "2D" or "3D".')
    if (dimension=="2D" and len(data[0])!=2) or (dimension=="3D" and len(data[0])!=3):
        raise ValueError('Data shape must match dimension!')
    if (label is not None) and len(data)!=len(label):
        raise ValueError('Number of rows in data must equal to length of label!')


    mpl.style.use(sty)

    # 2D scatter plot
    if dimension=="2D":

        # Plot with label
        if label is not None:
            label=np.array(label)
            lab=list(set(label))
            for index, l in enumerate(lab):
                if l==0:
                    xx='M41-N'
                    # xx='0'
                elif l==1:
                    xx="M41-P"
                    # xx="1"
                elif l==2:
                    xx = "H41-N"
                    # index = 9
                    # xx = "2"
                elif l==3:
                    xx = "H41-P"
                    # index = 10
                    # xx = "3"
                plt.scatter(data[label==l,0], data[label==l,1],
                                c='C{!r}'.format(index),
                                s=point_size, label=xx,
                                alpha=alpha)
                plt.xticks(weight='bold')
                plt.yticks(weight='bold')

            plt.legend(prop={'weight':'bold'},**kwargs)
            # 添加x轴和y轴标签
            # plt.ylabel(u'Negative Sampels', fontsize=20)
            # plt.ylabel(u'positive Sampels', fontsize=20)
            # plt.xlabel(u'Dimensions', fontsize=20)

            # plt.legend(fontsize=15,fontweigh='bold')
        # Plot without label
        else:
            plt.scatter(data[:,0], data[:,1], s=point_size, alpha=alpha)

        if aes_label is not None:
                plt.xlabel(aes_label[0])
                plt.ylabel(aes_label[1])


    # 3D scatter plot
    if dimension=="3D":
        splot = plt.subplot(111, projection='3d')

        # Plot with label
        if label is not None:
            label=np.array(label)
            lab=list(set(label))

            for index, l in enumerate(lab):
                splot.scatter(data[label==l,0], data[label==l,1], data[label==l,2],
                              s=point_size,
                              color='C{!r}'.format(index),
                              label=l)
            plt.legend(**kwargs)
        # Plot without label
        else:
            splot.scatter(data[:,0], data[:,1], data[:,2],s=point_size)

        if aes_label is not None:
            splot.set_xlabel(aes_label[0])
            splot.set_ylabel(aes_label[1])
            splot.set_zlabel(aes_label[2])

    plt.title(title,fontweight='bold')
    # plt.savefig('./tsne_H41.tif',dpi=350)
    plt.savefig('./tsne_H41_model_all.tif',dpi=350)
    # plt.savefig('./tsne_S51.tif',dpi=350)

    plt.show()


def draw(tsne,emb):
    if tsne:
        tsne = manifold.TSNE(2)
        rawdata = emb.iloc[:,1:]
        pc = tsne.fit_transform(rawdata)
        scatter(pc,dimension="2D",label=emb.label,title='H41-model')
    else:
        pca = decomposition.PCA(2)
        rawdata = emb.iloc[:,1:]
        pc = pca.fit_transform(rawdata)
        scatter(pc,dimension="2D",label=emb.label,title='pca',  aes_label=['PCA-1','PCA-2'])

if __name__=="__main__":
    # emb=pd.read_csv('Features/H41/Train_Test_feature.csv',header=None)
    # emb=pd.read_csv('Features/S51/Train_Test_feature9.csv',header=None)
    # emb=pd.read_csv('M41_H41_feature_all_max_min.csv',header=None)
    emb=pd.read_csv('H41model_All_feature.csv',header=None)
    emb=emb.rename(columns={0:"label"})
    # print(emb.label)
    #第一列为label
    # draw(True,emb)
    draw(True,emb)
