#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:59:26 2022

@author: vladislav
"""
import pandas
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv('./adult.csv', delimiter=(','))

labelEncoder = LabelEncoder()
labelEncoder.fit(data['income'])
transformedIncome = pandas.Series(data = labelEncoder.transform(data['income']))
data['income'] = transformedIncome

riches = data[data.income > 0]
riches = riches.reset_index()

hashmap = dict()
for index, row in riches.iterrows():
    if hashmap.get(row['age']) == None:
        hashmap[row['age']]=1
    else:
        hashmap[row['age']]=hashmap.get(row['age'])+1

plt.bar(hashmap.keys(), hashmap.values())
plt.title("Ages to Income count (rich)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()