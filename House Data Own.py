# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:57:33 2022

@author: satap
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor

#file Import

df=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
"""X = df.iloc[:,0:80]
y = df.iloc[:,80]

#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(1,80))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")"""


df.drop(['EnclosedPorch','KitchenAbvGr','BsmtHalfBath','LowQualFinSF','BsmtFinSF2','OverallCond','MSSubClass'],axis=1,inplace=True)
df_test.drop(['EnclosedPorch','KitchenAbvGr','BsmtHalfBath','LowQualFinSF','BsmtFinSF2','OverallCond','MSSubClass'],axis=1,inplace=True)

df['LotFrontage']= df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)
df['MasVnrType']= df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea']= df['MasVnrArea'].fillna(df['MasVnrArea'].mean())
df['BsmtQual']= df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtCond']= df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtExposure']= df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType1']= df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtFinType2']= df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['Electrical']= df['Electrical'].fillna(df['Electrical'].mode()[0])
df['FireplaceQu']= df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']= df['GarageType'].fillna(df['GarageType'].mode()[0])
df['GarageFinish']= df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']= df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']= df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df['GarageYrBlt']= df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
df_test.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df_test.drop(['Alley'],axis=1,inplace=True)
df_test['MSZoning']= df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['LotFrontage']= df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['Utilities']= df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])
df_test['Exterior1st']= df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])
df_test['Exterior2nd']= df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])
df_test['MasVnrType']= df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])
df_test['MasVnrArea']= df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())
df_test['BsmtQual']= df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtCond']= df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['BsmtExposure']= df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['BsmtFinType1']= df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])
df_test['BsmtFinType2']= df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_test['BsmtFinSF1']= df_test['BsmtFinSF1'].fillna(df_test['BsmtFinSF1'].mean())
df_test['BsmtUnfSF']= df_test['BsmtUnfSF'].fillna(df_test['BsmtUnfSF'].mean())
df_test['TotalBsmtSF']= df_test['TotalBsmtSF'].fillna(df_test['TotalBsmtSF'].mean())
df_test['BsmtFullBath']= df_test['BsmtFullBath'].fillna(df_test['BsmtFullBath'].mean())
df_test['KitchenQual']= df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])
df_test['Functional']= df_test['Functional'].fillna(df_test['Functional'].mode()[0])
df_test['FireplaceQu']= df_test['FireplaceQu'].fillna(df_test['FireplaceQu'].mode()[0])
df_test['GarageType']= df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['GarageYrBlt']= df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mean())
df_test['GarageFinish']= df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['GarageCars']= df_test['GarageCars'].fillna(df_test['GarageCars'].mean())
df_test['GarageArea']= df_test['GarageArea'].fillna(df_test['GarageArea'].mean())
df_test['GarageQual']= df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])
df_test['GarageCond']= df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test['SaleType']= df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])


"""df_test.info()
X = df.iloc[:,0:69]
y = df.iloc[:,-1]"""

#Catogorical 
cols = df.columns
num_cols = df._get_numeric_data().columns
num_column=num_cols.to_frame(index=False)
catogorical_data=list(set(cols) - set(num_cols))

main_df=df.copy()
main_df_test=df_test.copy()
df=pd.concat([df,df_test],axis=0)
df_merge = pd.get_dummies(data=df, columns=catogorical_data)

# Sales Column to move to last
cols = list(df_merge.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('SalePrice')) #Remove SalePrice from list

df_merge = df_merge[cols+['SalePrice']]
"""df.drop(catogorical_data,axis=1,inplace=True)

df_merge=pd.merge(left=df_dummies,right=df,on='Id')
df_merge=df_merge.loc[:,~df_merge.columns.duplicated()].copy()"""

#Test Train Split

X_train=df_merge.iloc[:1460,1:-1]
y_train=df_merge.iloc[:1460,-1]
X_test=df_merge.iloc[1460:,1:-1]
y_test=df_merge.iloc[1460:,-1]

# Standar Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)

X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)


#PCA

from sklearn.decomposition import PCA

# create the PCA object
# the number of components chosen will be the new number of features!

pca = PCA(n_components = 40, random_state=0)

# fit the PCA model to the data

pca.fit(X_scaled)

# it's like we have 40 new axis (those defined by the PCA principal components)

X_pca = pca.transform(X_scaled)
X_pca = pd.DataFrame(X_pca)

# Grid Search goes through all combinations of hyperparameters
from sklearn.model_selection import GridSearchCV

# we need to define what we consider the "full list" of hyperparameters

# Number of trees in random forest
n_estimators = [10,100,500]
# Mximum number of total leaves to consider
max_leaf_nodes = [15, 30, 40,50,60]
# Maximum number of levels in each tree
max_depth = [5,6,7,10]

# Create the  grid 
# this is a dictionary from hyperparameters to potential values
# the keys in this dictionary have to match the names of the hyperparameters in the documentation of the model

"""grid = {'n_estimators': n_estimators,
        'max_leaf_nodes': max_leaf_nodes,
        'max_depth': max_depth}
rfc=RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator = rfc , param_grid = grid, cv = 5) # the CV is the crossvalidation splitting strategy.
grid_search.fit(X_train, y_train) # training all the combinations.
d=grid_search.best_params_"""

forest = RandomForestRegressor(n_estimators=500, # the best parameters
                               max_leaf_nodes=100, 
                               max_depth=50, 
                               random_state=1)


forest.fit(X_pca, y_train) # training our
# create object"""

scaler = StandardScaler()

# fit

scaler.fit(X_test)

# transform 

X_scaled_test = scaler.transform(X_test)

X_scaled_test = pd.DataFrame(X_scaled_test, columns=X_test.columns)

print('Train score:', forest.score(X_pca,y_train))                                                            

# create the PCA object
# the number of components chosen will be the new number of features!

pca = PCA(n_components = 40, random_state=0)

# fit the PCA model

pca.fit(X_scaled_test)

# it's like we have three new axis (those defined by the PCA principal components)

Xp_test = pca.transform(X_scaled_test)
submission = forest.predict(Xp_test)
y_pred=pd.DataFrame(submission)
print(forest.score(Xp_test,y_pred))
df_test['SalePrice'] = submission
final_submission = df_test[['Id','SalePrice']]
final_submission.to_csv('submission2.csv', index=False)


