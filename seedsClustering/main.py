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

K=3
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


for i in set(dbscan.labels_):
    plt.scatter(pcaMapping[dbscan.labels_ == i][:, 0],
                pcaMapping[dbscan.labels_ == i][:, 1])
plt.title('DBSCAN Results')
plt.show()
plt.clf()

'''
Our experiments showed that KMeans and DBSCAN yields similar results with the following parameters
KMeans(n_clusters=3)
DBSCAN(eps=1, min_samples=3)

Compare the results
'''

def clusterIntersection(cluster1, cluster2):
    result = 0
    for c1 in cluster1:
        for c2 in cluster2:
            if (c1 - c2).sum() == 0:
                result += 1
                continue
    return result

#clusterIntersection(seeds[kmeans.labels_ == 0].values,seeds[kmeans.labels_ == 0].values)

for kmeansIndex in set(kmeans.labels_):
    for dbscanIndex in set(dbscan.labels_):
        print(kmeansIndex,',',dbscanIndex,',',
        clusterIntersection(
            seeds[kmeans.labels_ == kmeansIndex].values,
            seeds[dbscan.labels_ == dbscanIndex].values))
