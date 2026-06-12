import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from types import SimpleNamespace
from warnings import warn


def CovPCA(data, CovMat, ncomp = 2, values = True):

    eigenval, eigenvect = np.linalg.eigh(CovMat)
    
    idx = np.argsort(eigenval)[::-1]
    eigenval = eigenval[idx]
    eigenvect = eigenvect[:, idx]
    
    base = eigenvect[:,0:ncomp]
    neigenval = eigenval[0:ncomp]
    info = (neigenval / sum(eigenval) * 100).round(2)
    axe1_info = info[0]
    axe2_info = info[1]

    ProjMat = np.dot(base, base.T)

    proj = np.dot(data, ProjMat)

    if (values == True):
        return SimpleNamespace(projection = proj, pct_info = info, projection_base = base, eigenvalues = neigenval)

    if (values == False and ncomp == 2):
        fig, axes = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x = proj[0,:],
                        y = proj[1,:])
        axes.set_xlabel(f"1st component | {axe1_info} %")
        axes.set_ylabel(f"2nd component | {axe2_info} %")

    else : 
        raise ValueError("values has to be a Boolean or ncomp = 2 if value = False")

    