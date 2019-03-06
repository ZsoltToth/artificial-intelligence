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

