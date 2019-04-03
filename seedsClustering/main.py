'''
Clustering example over the Seeds Dataset from UCI Machine Learning Repo.

http://archive.ics.uci.edu/ml/datasets/seeds

1. area A,
2. perimeter P,
3. compactness C = 4*pi*A/P^2,
4. length of kernel,
5. width of kernel,
6. asymmetry coefficient
7. length of kernel groove.
'''

import pandas as pd

seeds = pd.read_csv('data/seeds_dataset.txt',
                    sep='\t',
                    names=['area',
                           'perimeter',
                           'compactness',
                           'kernelLength',
                           'kernelWidth',
                           'asymmetryCoeff',
                           'kernelGroove'])

seeds = seeds.dropna()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(seeds)

import matplotlib.pyplot as plt

plt.scatter(pca.transform(seeds)[:,0],pca.transform(seeds)[:,1])
plt.show()