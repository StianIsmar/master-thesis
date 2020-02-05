import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
digits = datasets.load_digits()
import pandas as pd

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

X = pd.DataFrame(X, columns=['one','two'])
print(X.shape)
for i, col in enumerate(X.columns):
    print(X[col].nunique())
    if X[col].nunique() < 9:
        X.drop(col,axis =1, inplace = True)

print(X.shape)

'''
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)
'''