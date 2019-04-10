'''
Train a Classifier for the Credit Approval Dataset

http://archive.ics.uci.edu/ml/datasets/Credit+Approval
'''

import pandas as pd

dataset = pd.read_csv('data/crx.data',names=['A'+str(i+1) for i in range(0,16)])
dataset['A1'] = dataset['A1'].astype('category')
dataset = dataset[dataset['A2'] != '?']
dataset['A2'] = dataset['A2'].astype('float64')
dataset['A4'] = dataset['A4'].astype('category')
dataset['A5'] = dataset['A5'].astype('category')
dataset['A6'] = dataset['A6'].astype('category')
dataset['A7'] = dataset['A7'].astype('category')

dataset['A9'] = dataset['A9'].astype('category')
dataset['A10'] = dataset['A10'].astype('category')

dataset['A12'] = dataset['A12'].astype('category')
dataset['A13'] = dataset['A13'].astype('category')
dataset = dataset[dataset['A14'] != '?']
dataset['A14'] = dataset['A14'].astype('int64')

dataset['A16'] = dataset['A16'].astype('category')



X = dataset.drop('A16',axis=1)
cat_columns = X.select_dtypes(['category']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
Y = dataset['A16'].cat.codes

from sklearn import tree
from sklearn.neural_network import MLPClassifier

decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(X,Y)

mlp = MLPClassifier(
    hidden_layer_sizes=(5,7,5,3),
    activation='tanh',
    solver='sgd')

mlp.fit(X,Y)

#pd.DataFrame(decisionTree.predict(X) == Y).groupby(0).size()

