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

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

decisionTree = tree.DecisionTreeClassifier()
decisionTree.fit(X_train,Y_train)

mlp = MLPClassifier(
    hidden_layer_sizes=(5,7,5,3),
    activation='tanh',
    solver='sgd')

mlp.fit(X_train,Y_train)
'''
#pd.DataFrame(decisionTree.predict(X) == Y).groupby(0).size()

pd.DataFrame(mlp.predict(X_test) == Y_test).groupby(0).size()
pd.DataFrame(decisionTree.predict(X_test) == Y_test).groupby(0).size()

mlp.score(X_test,Y_test)
0.6526946107784432
decisionTree.score(X_test,Y_test)
0.8023952095808383
'''

from sklearn.model_selection import cross_val_score

scoreMLP = cross_val_score(mlp, X,Y,cv=5)
scoreDT = cross_val_score(decisionTree, X,Y,cv=5)

#Test 3 different KNN Classifiers
from sklearn.neighbors import KNeighborsClassifier

knn = [KNeighborsClassifier(n_neighbors=i) for i in {3,5,7} ]
scoreKNN = [cross_val_score(knn_i, X,Y,cv=5) for knn_i in knn]

#GridSearchCV
from sklearn.model_selection import GridSearchCV

mlpParameters = {
    'hidden_layer_sizes':[(3,4,5),(10,20,10),(10,5,2)],
    'activation':('tanh','relu','logistic'),
    'solver':('sgd','adam'),
    'learning_rate':('constant','adaptive')
}
mlp = MLPClassifier()
gridSearch = GridSearchCV(mlp,mlpParameters, cv=3)
gridSearch.fit(X,Y)
