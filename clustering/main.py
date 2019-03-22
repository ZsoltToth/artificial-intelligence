'''
Some quick warm up exercises for part time students
'''
'''
def fact(n):
    if(n <= 1):
        return 1
    else:
        return n * fact(n-1)

print(fact(4))
print([fact(i) for i in range(5)])

import numpy as np

a = [
    [1,-2,3],
    [2,1,1],
    [-3,2,-2]
]

b = [7,4,-10]
x = np.linalg.solve(a,b)
print(x)
print(a @ x)

del a, b, x

x = np.linspace(0,10,1e3)
y = np.sin(x)

for i in range(0,x.size):
    print(x[i],y[i])

import matplotlib.pyplot as plt

plt.plot(x,y)
plt.show()

'''
import matplotlib.pyplot as plt
import pandas as pd

hun = pd.read_excel('./data/hungary.xls',
                    skiprows=[0,1,2],
                    sheet_name='Data')

#Selec a row based on the value of a column
mrchExp = hun[hun['Indicator Code'] == 'TX.VAL.MRCH.CD.WT']
mrchExp = mrchExp.drop(['Country Name', 'Country Code',
                        'Indicator Name', 'Indicator Code'], axis=1)

mrchExp = mrchExp.T.reset_index()
mrchExp = mrchExp.rename(columns={mrchExp.columns[0] : 'years',
                                  mrchExp.columns[1] : 'export'})

plt.clf()
plt.plot(mrchExp['years'],mrchExp['export'])
plt.title('Merchandise Export of Hungary US$')
#plt.savefig('merhandise.png')
plt.show()
del mrchExp


SELECTED_INDICATORS = ['TX.VAL.MRCH.CD.WT','MS.MIL.XPND.CN','MS.MIL.MPRT.KD']
hunSelectedFeatures = hun[hun['Indicator Code'].isin(SELECTED_INDICATORS)]
hunSelectedFeatures = hunSelectedFeatures.drop(['Country Name', 'Country Code',
                        'Indicator Name'], axis=1)


hunSelectedFeatures = hunSelectedFeatures.set_index('Indicator Code').T.reset_index()

plt.clf()
for i in SELECTED_INDICATORS:
    plt.plot(hunSelectedFeatures['index'],hunSelectedFeatures[i])
plt.show()

plt.clf()
plt.imshow(hunSelectedFeatures.corr())
plt.show()

del hunSelectedFeatures, SELECTED_INDICATORS

'''
IT.NET.USER.ZS
IT.NET.SECR.P6
IT.NET.SECR
IT.NET.BBND.P2
IT.NET.BBND
IT.MLT.MAIN.P2
IT.MLT.MAIN
IT.CEL.SETS.P2
IT.CEL.SETS
'''

SELECTED_INDICATORS = [
'IT.NET.USER.ZS',
'IT.NET.SECR.P6',
'IT.NET.SECR',
'IT.NET.BBND.P2',
'IT.NET.BBND',
'IT.MLT.MAIN.P2',
'IT.MLT.MAIN',
'IT.CEL.SETS.P2',
'IT.CEL.SETS'
]

hunIT = hun[hun['Indicator Code'].isin(SELECTED_INDICATORS)]
hunIT= hunIT.drop(['Country Name', 'Country Code',
                        'Indicator Name'], axis=1)


hunIT = hunIT.set_index('Indicator Code').T.reset_index()

plt.clf()
for i in SELECTED_INDICATORS:
    plt.plot(hunIT['index'],hunIT[i])
plt.show()

plt.clf()
plt.imshow(hunIT.corr())
plt.show()

'''
Scaling and PCA
'''


from sklearn.preprocessing import StandardScaler
hunITFiltered = hunIT.dropna()
scaledHunIT = StandardScaler().fit_transform(hunITFiltered.drop(['index'], axis=1))

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pcaDF = pd.DataFrame(data= pca.fit_transform(scaledHunIT),columns=['PC1','PC2'])
pd.concat([pcaDF,pd.DataFrame(SELECTED_INDICATORS)],axis=1)

plt.clf()
fig = plt.scatter(pcaDF['PC1'],pcaDF['PC2'])
plt.title('2D PCA Projection of IT Changes of Hungary between 2010-2017')
plt.xlabel('PC1')
plt.ylabel('PC2')
for i in range(hunITFiltered['index'].values.size):
    plt.annotate(hunITFiltered['index'].values[i],(pcaDF['PC1'][i],pcaDF['PC2'][i]))
plt.show()

'''
Clustering
'''

from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram
hunITFilteredDistanceMatrix = pdist(hunITFiltered.drop('index',axis=1).values)

z = single(hunITFilteredDistanceMatrix)
d = dendrogram(z)

plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(hunITFiltered.drop('index',axis=1).values)