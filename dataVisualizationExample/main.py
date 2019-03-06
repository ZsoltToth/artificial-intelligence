import pandas as pd

countriesMetadata = pd.read_excel(
    'data/worldbankPopulation1564.xls',
    sheet_name='Metadata - Countries')

population = pd.read_excel(
    'data/worldbankPopulation1564.xls',
    sheet_name='Data',
    skiprows=[0,1,2])
