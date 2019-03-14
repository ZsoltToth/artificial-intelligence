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
grpByCountries = population.drop(['Country Name','Indicator Name','Indicator Code'],axis=1).melt('Country Code').pivot(index='variable',columns='Country Code', values='value')
'''
Question: In which countires changed the population similarly?
i) Correlation Matrix
'''
corr = grpByCountries.corr().abs()
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

'''
