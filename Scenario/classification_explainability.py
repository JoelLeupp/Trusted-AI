# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:54:37 2021

@author: Besitzer
"""

import numpy as np
import pickle
import statistics

from xgboost import XGBClassifier
from sympy.core.numbers import igcd
from datetime import datetime

from sklearn import preprocessing
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import MinMaxScaler,QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random

#%%

os.chdir(r"C:\Users\Besitzer\Documents\MEGA\Master Project\Trusted AI\Scenario\Models")
# Data load
df_0=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_0.csv")
df_1=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_1.csv")
df_2=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_2.csv")
df_3=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_3.csv")
df_4=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_4.csv")
df_5=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_5.csv")
df_6=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_6.csv")
df_mix=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_normal_all.csv")

# Attacks
df_fakepsd=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_fakePSD.csv")
df_sendout=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_sendOut.csv")
df_write=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_write.csv")
df_random=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_random.csv")
df_exchange=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_exchange.csv")
df_hide=pd.read_csv("./Datasets/Rp3_3rd_capture/rp3_hide.csv")

#%%

# Label vectors according to the timestamps

df_0.loc[(df_0['timestamp'] >= 1615419435000) & (df_0['timestamp'] <= 1615423051000), 'label'] = 1
df_1.loc[(df_1['timestamp'] >= 1615925651000) & (df_1['timestamp'] <= 1615929271000), 'label'] = 2
df_2.loc[(df_2['timestamp'] >= 1615930017000) & (df_2['timestamp'] <= 1615933591000), 'label'] = 3
df_3.loc[(df_3['timestamp'] >= 1615969307000) & (df_3['timestamp'] <= 1615972800000), 'label'] = 4
df_4.loc[(df_4['timestamp'] >= 1615974874000) & (df_4['timestamp'] <= 1615978500000), 'label'] = 5
df_5.loc[(df_5['timestamp'] >= 1615978899000) & (df_5['timestamp'] <= 1615982700000), 'label'] = 6
df_6.loc[(df_6['timestamp'] >= 1615989015000) & (df_6['timestamp'] <= 1615992622000), 'label'] = 7
df_mix.loc[(df_mix['timestamp'] >= 1615996500000) & (df_mix['timestamp'] <= 1616000131000), 'label'] = 8


# Label each df
df_fakepsd.loc[(df_fakepsd['timestamp'] >= 1615452439000) & (df_fakepsd['timestamp'] <= 1615456051000), 'label'] = 9
df_write.loc[(df_write['timestamp'] >= 1615834554000) & (df_write['timestamp'] <= 1615838131000), 'label'] = 10
df_sendout.loc[(df_sendout['timestamp'] >= 1615795500000) & (df_sendout['timestamp'] <= 1615799131000), 'label'] = 11
df_random.loc[(df_random['timestamp'] >= 1615913848000) & (df_random['timestamp'] <= 1615917451000), 'label'] = 12
df_exchange.loc[(df_exchange['timestamp'] >= 1616585145000) & (df_exchange['timestamp'] <= 1616589463000), 'label'] = 13
df_hide.loc[(df_hide['timestamp'] >= 1616672277000) & (df_hide['timestamp'] <= 1616676574000), 'label'] = 14

df_0.dropna(subset=['label'], inplace=True)
df_1.dropna(subset=['label'], inplace=True)
df_2.dropna(subset=['label'], inplace=True)
df_3.dropna(subset=['label'], inplace=True)
df_4.dropna(subset=['label'], inplace=True)
df_5.dropna(subset=['label'], inplace=True)
df_6.dropna(subset=['label'], inplace=True)
df_mix.dropna(subset=['label'], inplace=True)
df_fakepsd.dropna(subset=['label'], inplace=True)
df_write.dropna(subset=['label'], inplace=True)
df_sendout.dropna(subset=['label'], inplace=True)
df_random.dropna(subset=['label'], inplace=True)
df_exchange.dropna(subset=['label'], inplace=True)
df_hide.dropna(subset=['label'], inplace=True)

#%%

# Concatenate dataframes 
#df_comb = pd.concat([df_exchange, df_hide])
df_comb = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_mix, df_fakepsd, df_write, df_sendout, df_random, df_exchange, df_hide])

# Remove features related to time
df_comb = df_comb.drop(["time","timestamp"], axis=1)

# Remove features with constant values
df_comb=df_comb.loc[:, (df_comb != df_comb.iloc[0]).any()]

correlation_matrix = df_comb.corr()

# feature list
feat_list=df_comb.columns

#%%

# train model

# Split dataset in X (features) and Y (labels)
df_X = df_comb.iloc[:,:-1]
df_Y = df_comb.iloc[:,-1: ]

# Split the datast in training and testing set
X_train, X_test, y_train, y_test = train_test_split(df_X,df_Y, test_size=0.20, random_state=42)

#Create a Random Forest classifier
rnd_clf = RandomForestClassifier()

#Cross validation
y_train_pred = cross_val_predict(rnd_clf, X_train, y_train, cv=5)

#Confusion matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)

# Train RF with the training set
rnd_clf.fit(X_train, y_train)

# Test with the testing set
y_pred_rf = rnd_clf.predict(X_test)

# Print Accuracy and confusion matrix
print(accuracy_score(y_test,y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

#%%
#Print the relevance of each feature

feat_labels=df_X.columns
importances=rnd_clf.feature_importances_
indices=np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

relevance = pd.DataFrame({"relevance":importances[indices]}, feat_labels[indices])
relevance["relevance"].plot.bar()

#%% 
remove highly correlated features

corr_matrix = df_comb.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_comb.drop(to_drop, axis=1, inplace=True)