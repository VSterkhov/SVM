#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:37:27 2022

@author: vladislav
"""

import numpy
import operator
import pandas
import scipy.stats
import seaborn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data = pandas.read_csv('./adult.csv', delimiter=(','))

labelEncoder = LabelEncoder()
labelEncoder.fit(data['income'])
transformedIncome = pandas.Series(data = labelEncoder.transform(data['income']))
data['income'] = transformedIncome
labelEncoder.fit(data['gender'])
transformedGender = pandas.Series(data = labelEncoder.transform(data['gender']))
data['gender'] = transformedGender

targetData = pandas.get_dummies(data.loc[:, ['age', 'hours-per-week', 'educational-num', 'gender', 'race', 'capital-gain', 'capital-loss']])
Y = data['income']

X_train, X_test, Y_train, Y_test = train_test_split(targetData, Y, test_size=0.2)

logRegModel = LogisticRegression(max_iter=1000)
logRegModel.fit(X_train, Y_train)
print('LogisticRegression Score: ', logRegModel.score(X_test,Y_test))

svcModel = SVC(gamma='auto')
svcModel.fit(X_train, Y_train)
print('SVM Score: ', svcModel.score(X_test,Y_test))
