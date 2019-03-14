import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

countriesMetadata = pd.read_excel(
    'data/worldbankPopulation1564.xls',
    sheet_name='Metadata - Countries')

population = pd.read_excel(
    'data/worldbankPopulation1564.xls',
    sheet_name='Data',
    skiprows=[0,1,2])


'''
Create a line chart which shows the population changes of Hungary between 1960 and 2017
'''

years = np.arange(start=1960, stop=2018,step=1)

hunPopulation = [population[population['Country Code'] == 'HUN']
                 [str(y)].values[0] for y in years]

plt.plot(years, hunPopulation)
plt.show()
del years, hunPopulation
'''
Task
Generate a line chart for each country. 
Title should be filled properly.

Bonus task:
Title should contain the Region.
'''

'''
Transform the dataset in order to the country codes became columns and the year becomes also a column 
'''

#grpByCountries = population.drop(['Country Name','Indicator Name','Indicator Code'],axis=1).groupby(by='Country Code').mean().reset_index()
#grpByCountries = population.drop(['Country Name','Indicator Name','Indicator Code'],axis=1).melt('Country Code').pivot(index='variable',columns='Country Code', values='value')
populationTransposed = population.drop(['Country Name','Indicator Name','Indicator Code','2018'],axis=1).set_index('Country Code').transpose();
'''
Question: In which countires changed the population similarly?
i) Correlation Matrix
'''
corr = populationTransposed.corr()
plt.imshow(corr);
plt.show()
'''
In which countries changed the population similarly to HUN
'''
corr['HUN'][corr.abs()['HUN'] > .8]

import seaborn as sns
sns.heatmap(corr.abs(),square=True)
plt.title("Correlation Matrix with Seaborn")
plt.show()

'''
Perform Principla Component Analysis and Visualize the results.

First we have to remove the records with missing data and scale the dataset. 
'''
popFiltered = population.drop(['Country Name','Indicator Name','Indicator Code','2018'],axis=1).dropna()
from sklearn.preprocessing import StandardScaler
countryCodes = popFiltered['Country Code']

scaledPop = StandardScaler().fit_transform(popFiltered.drop(['Country Code'], axis=1))
#This is the same
#scaler = StandardScaler().fit(popTFiltered)
#scaler.transform(popTFiltered)
'''
We define the number of Principal Components to 2. 
Hence, the data set can be mapped to a 2D Scatter Chart. 
'''
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

#Transforming the dataset into the Vector Space of the Principal Components
#pca.fit(scaledPop)
#pcaDF = pd.DataFrame(data= pca.transform(scaledPop),columns=['PC1','PC2'])
pcaDF = pd.DataFrame(data= pca.fit_transform(scaledPop),columns=['PC1','PC2'])
pd.concat([pcaDF,countryCodes],axis=1)
'''
Visualization of the Countries on the Scatter Chart
Scatter chart is annotated with the Country Codes
'''
plt.clf()
fig = plt.scatter(pcaDF['PC1'],pcaDF['PC2'])
plt.title('2D PCA Projection of Population Changes of Countries')
plt.xlabel('PC1')
plt.ylabel('PC2')
for i, cc in enumerate(countryCodes):
    plt.annotate(cc,(pcaDF['PC1'][i],pcaDF['PC2'][i]))
plt.show()
