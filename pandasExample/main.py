'''
data folder contains some data sets downloaded from https://data.worldbank.org/indicator

Data Sets are related to Science and Technology.

Data sets are the following:
 - Charges for the use of intellectual property, payments (BoP, current US$)
 - High-technology exports (% of manufactured exports)
 - Research and development expenditure (% of GDP)
 - Scientific and technical journal articles
 - Researchers in R&D (per million people)
 - Technicians in R&D (per million people)



'''

import pandas as pd

#xlrd package has to be installed because we are reading Excel files.
charges = pd.read_excel('data/chargesForUseOfIntellectualProperty.xls',
                        sheet_name='Data',
                        skiprows=[0,1,2])

HTExport = pd.read_excel('data/highTechExport.xls',
                        sheet_name='Data',
                        skiprows=[0,1,2])

journalArticles = pd.read_excel('data/journalArticles.xls',
                        sheet_name='Data',
                        skiprows=[0,1,2])

expenditures = pd.read_excel('data/RDExpenditure.xls',
                        sheet_name='Data',
                        skiprows=[0,1,2])

researchers = pd.read_excel('data/researchersInRD.xls',
                        sheet_name='Data',
                        skiprows=[0,1,2])

technicians = pd.read_excel('data/techniciansInRD.xls',
                        sheet_name='Data',
                        skiprows=[0,1,2])
'''
We can browse the content of these dataframes in PyCharm. 
Special Variables -> "View as DataFrame"

We can see that most of the data is missing.
Missing values are denoted by NaN and they can be counted by the isnull and sum functions.
The following line calculates the ratio of the missing values in the expenditures data set.

expenditures.isnull().sum().sum() / expenditures.size

Due to the high missing ratio, we narrow our search after 2005.
In addition, we drop the records which contain missing data in any year. 
'''

SELECTED_PERIOD = [
    'Country Name',
    'Country Code',
    '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'
    ]



charges = charges[SELECTED_PERIOD].dropna()
HTExport = HTExport[SELECTED_PERIOD].dropna()
journalArticles = journalArticles[SELECTED_PERIOD].dropna()
expenditures = expenditures[SELECTED_PERIOD].dropna()
researchers = researchers[SELECTED_PERIOD].dropna()
technicians = technicians[SELECTED_PERIOD].dropna()
'''
Because the technicians data frame does not count much records after reduction, we considers only the other data sets.
'''
del technicians

countryCodes = set(charges['Country Code']).intersection(
    HTExport['Country Code']).intersection(
    journalArticles['Country Code']).intersection(
    expenditures['Country Code']).intersection(
    researchers['Country Code'])
#The following Country Codes denote areas so they are removed
countryCodes.remove('EAS')
countryCodes.remove('ECS')
countryCodes.remove('EMU')
countryCodes.remove('LTE')

'''
At this point we have 5 data sets (indicators) which contains time series (2005-2016) about 26 countries (countryCodes).
There is a choice now. 
a) We can analyze each indicator separately.
b) We can analyze the countries by the indicators.  

This example focuses on pandas so lets continue with option b.

So lets query the dataset for a single country (GBR) and merge these indicators into a single data set.

Expected result
Year Charges Export Expenditures Journals Researchers
2005
2006
etc.
'''
#First, the dataframe has to be composed.
#We simply query GBR from our separate data sets and concate the results.
gbr = pd.concat([
    charges[charges['Country Code'] == 'GBR'],
    HTExport[HTExport['Country Code'] == 'GBR'],
    expenditures[expenditures['Country Code'] == 'GBR'],
    journalArticles[journalArticles['Country Code'] == 'GBR'],
    researchers[researchers['Country Code'] == 'GBR']
])
#We have to insert a column which denote the indicators
gbr.insert(loc=0, column='Indicator',value=['Charges','Export','Expenditure','Journals','Researchers'])
#Then we drop the Country Name and Code because they became useless
gbr = gbr.drop(['Country Name','Country Code'],axis=1)
#Next, we set the recently added column as index and transpose the data frame
#The axis is renamed to Years and the index is reseted.
gbr = gbr.set_index('Indicator').T.rename_axis("Years").reset_index()

corr = gbr.drop('Years',axis=1).corr().abs()

import matplotlib.pyplot as plt

plt.plot(gbr['Years'],gbr['Charges'],label='Charges')
plt.plot(gbr['Years'],gbr['Export'],label='Export')
plt.plot(gbr['Years'],gbr['Expenditure'],label='Expenditure')
plt.plot(gbr['Years'],gbr['Journals'],label='Journals')
plt.plot(gbr['Years'],gbr['Researchers'],label='Researchers')
plt.show()

from sklearn.preprocessing import StandardScaler