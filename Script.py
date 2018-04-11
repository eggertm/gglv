import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# TODO: Add all code belonging to the report here

class Data:
    #storage class for data
    def __init__(self, filename):
        x = pd.read_csv(filename,index_col='id')
        y = x['loss'].values

        categoryColumnNames = [c for c in x.columns if c.startswith('cat')]
        convertedX = pandas.get_dummies(x,drop_first=True,columns = categoryColumnNames)
        goodColumnNames = [l for l in convertedX.columns if l not in ('id','loss')]
        X = convertedX.loc[:,goodColumnNames].values
        self.X = X
        self.y = y
    
    def GetSplit(self, test_size=0.20, pcaComponents = None):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size)
        if pcaComponents is not None:
            pca = PCA(n_components = pcaComponents)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        return  X_train, X_test, y_train, y_test
    
    
