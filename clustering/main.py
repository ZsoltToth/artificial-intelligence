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

SELECTED_INDICATORS = ['TX.VAL.MRCH.CD.WT','MS.MIL.XPND.CN','MS.MIL.MPRT.KD']
hunSelectedFeatures = hun[hun['Indicator Code'].isin(SELECTED_INDICATORS)]
hunSelectedFeatures = hunSelectedFeatures.drop(['Country Name', 'Country Code',
                        'Indicator Name'], axis=1)

hunSelectedFeatures = hunSelectedFeatures.set_index('Indicator Code').T.reset_index()

plt.clf()
for i in SELECTED_INDICATORS:
    plt.plot(hunSelectedFeatures['index'],hunSelectedFeatures[i])
plt.show()