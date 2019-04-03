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

#Read the data set
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

#Remove records containing nan.
seeds = seeds.dropna()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(seeds)
#Visualize PCA Results
import matplotlib.pyplot as plt
pcaMapping = pca.transform(seeds)
plt.scatter(pcaMapping[:,0],pcaMapping[:,1])
plt.title('PCA')
plt.show()
plt.clf()

#Cluster the seeds with KMeans algorithm
from sklearn.cluster import KMeans

K=5
kmeans = KMeans(n_clusters=K).fit(seeds)
#Visualize the clusters

'''
plt.scatter(pcaMapping[kmeans.labels_==0][:,0],
            pcaMapping[kmeans.labels_ == 0][:, 1],
            marker='.')
plt.scatter(pcaMapping[kmeans.labels_==1][:,0],
            pcaMapping[kmeans.labels_ == 1][:, 1],
            marker='o')
plt.scatter(pcaMapping[kmeans.labels_==2][:,0],
            pcaMapping[kmeans.labels_ == 2][:, 1],
            marker='v')
'''
for i in range(0,K):
    plt.scatter(pcaMapping[kmeans.labels_ == i][:, 0],
                pcaMapping[kmeans.labels_ == i][:, 1])
plt.title('K-Means Results')
plt.show()
plt.clf()
#Cluster the seeds with DBSCAN Algorithm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(seeds)
dbscan = DBSCAN(eps=1, min_samples=3).fit(scaler.transform(seeds))


for i in range(0,K):
    plt.scatter(pcaMapping[dbscan.labels_ == i][:, 0],
                pcaMapping[dbscan.labels_ == i][:, 1])
plt.title('DBSCAN Results')
plt.show()
plt.clf()