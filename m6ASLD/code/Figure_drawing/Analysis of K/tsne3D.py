# %load ./tsne.py
from sklearn import decomposition, manifold
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import umap

try:
    import plotly.express as px
except:
    px = None


def scatter(data, dimension="2D", point_size=5, sty='default',
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
    if (dimension == "2D" and len(data[0]) != 2) or (dimension == "3D" and len(data[0]) != 3):
        raise ValueError('Data shape must match dimension!')
    if (label is not None) and len(data) != len(label):
        raise ValueError('Number of rows in data must equal to length of label!')

    mpl.style.use(sty)

    # 2D scatter plot
    if dimension == "2D":
        # Plot with label
        if label is not None:
            label = np.array(label)
            lab = list(set(label))
            for index, l in enumerate(lab):
                if index == 0:
                    index = '#87CEFA'
                else:
                    index = '#FFB6C1'
                plt.scatter(data[label == l, 0], data[label == l, 1],
                            c='{}'.format(index),
                            s=point_size, label=l,
                            alpha=alpha)
            plt.legend(**kwargs)
        # Plot without label
        else:
            plt.scatter(data[:, 0], data[:, 1], s=point_size, alpha=alpha)

        if aes_label is not None:
            plt.xlabel(aes_label[0])
            plt.ylabel(aes_label[1])

    # 3D scatter plot
    if dimension == "3D":
        splot = plt.subplot(111, projection='3d')

        # Plot with label
        if label is not None:
            label = np.array(label)
            lab = list(set(label))

            for index, l in enumerate(lab):
                splot.scatter(data[label == l, 0], data[label == l, 1], data[label == l, 2],
                              s=point_size,
                              color='C{!r}'.format(index),
                              label=l)
            plt.legend(**kwargs)


        # Plot without label
        else:
            splot.scatter(data[:, 0], data[:, 1], data[:, 2], s=point_size)

        if aes_label is not None:
            splot.set_xlabel(aes_label[0])
            splot.set_ylabel(aes_label[1])
            splot.set_zlabel(aes_label[2])
        splot.set_xlim(-200, 200)
        splot.set_ylim(-200, 200)
        splot.set_zlim(-200, 200)


        # splot.legend(prop={'weight': 'bold', 'size': 15})
        # plt.xticks(weight='bold', fontsize=15)
        # plt.yticks(weight='bold', fontsize=15)
        # plt.zticks(weight='bold', fontsize=15)

    plt.title(title)

    # plt.savefig('./uma.svg')
    # plt.savefig('./Global representations.tif',bbox_inches='tight',dpi=350)
    # plt.savefig('./Local representations 1.tif',bbox_inches='tight',dpi=350)
    plt.show()

def draw(method, emb):
    if method == 'tsne':
        tsne = manifold.TSNE(2)
        rawdata = emb.iloc[:, 1:]
        pc = tsne.fit_transform(rawdata)
        scatter(pc, dimension="3D", label=emb.label, aes_label=['tSNE-1', 'tSNE-2'])
        
    elif method == 'pca':
        pca = decomposition.PCA(2)
        rawdata = emb.iloc[:, 1:]
        pc = pca.fit_transform(rawdata)
        scatter(pc, dimension="2D", label=emb.label, title='pca', aes_label=['PCA-1', 'PCA-2'])
    else:
        reducer = umap.UMAP(random_state=42)
        rawdata = emb.iloc[:, 1:]
        pc = reducer.fit_transform(rawdata)
        scatter(pc, dimension="2D", label=emb.label, aes_label=['UMAP-1', 'UMAP-2'])

def draw_3d(method, emb):
    if method == 'tsne':
        tsne = manifold.TSNE(3)
        rawdata = emb.iloc[:, 1:]
        pc = tsne.fit_transform(rawdata)
        # scatter(pc, dimension="3D", label=emb.label, aes_label=['tSNE-1', 'tSNE-2','tSNE-3'])
        scatter(pc, dimension="3D", label=emb.label)

    elif method == 'pca':
        pca = decomposition.PCA(3)
        rawdata = emb.iloc[:, 1:]
        pc = pca.fit_transform(rawdata)
        scatter(pc, dimension="3D", label=emb.label, title='pca', aes_label=['PCA-1', 'PCA-2','PCA-3'])
    else:
        reducer = umap.UMAP(random_state=42,n_components=3)
        rawdata = emb.iloc[:, 1:]
        pc = reducer.fit_transform(rawdata)
        scatter(pc, dimension="3D", label=emb.label, aes_label=['UMAP-1', 'UMAP-2','UMAP-3'])

if __name__=="__main__":
    emb = pd.read_csv('Global_4_test.csv', header=None)
    # emb = pd.read_csv('Local_1_test.csv', header=None)
    emb = emb.rename(columns={0: "label"})
    # print(emb.label)
    # 第一列为label
    draw_3d('tsne', emb)
    # draw('tsne', emb)

