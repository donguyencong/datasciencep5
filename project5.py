#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:54:42 2017

@author: congdonguyen

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.ensemble as em
import sklearn.linear_model as lm


#Read data, clean data
data = pd.read_table("data-subset.tsv")
data = data.replace("-", np.NaN)
data = data.dropna(how="any")
data["OGrade"] = data["OGrade"].astype(int)

#Plot histogram
plt.figure(1)
maleSubjects = data.loc[data["Gender"] == "M"]
maleSubjects["Age"].plot.hist(stacked=True, legend=True, label="Male")
femaleSubjects = data.loc[data["Gender"] == "F"]
femaleSubjects["Age"].plot.hist(stacked=True, title="Age distribution", alpha = 0.75, label="Female", legend=True).set_ylabel("# of people")
plt.xlabel('Age')
plt.savefig("plot/hist.png")
plt.close()

#Plot scatter
plt.figure(2)
plt.title('Generosity vs Age')
plt.xlabel('Generosity')
plt.ylabel('Age')
data["generosity"] = data["IGrade"] - data["OGrade"]
data["generosity"] = data["generosity"] + np.random.rand(len(data))
data["Age"] = data["Age"] - np.random.rand(len(data))
plt.scatter(data["generosity"], data["Age"], c=["r","b"], alpha=0.6)
plt.savefig("plot/scatter.png")
plt.close()

#Seed
np.random.seed(len(data))
selection = np.random.binomial(1,.7,size=len(data)).astype(bool)
training = data[selection]

#RFC, Logistic Regression
rfc = em.RandomForestClassifier()
regr = lm.LogisticRegression(C=1.0, penalty='l1')

#Dummies variable
training["Gender"] = pd.get_dummies(training["Gender"])

#Model
X = training[["Age","Gender", "IGrade"]]

#Fit model
regr.fit(X, training["OGrade"])
rfc.fit(X, training["OGrade"])

#Get score
rfcScore = rfc.score(X, training["OGrade"])
print("Random Forest Classifier score is: " + str(rfcScore))
regrScore = regr.score(X, training["OGrade"])
print("Logistic Regression Score is " + str(regrScore))

#Confusion matrix
regrConfusionMatrix = confusion_matrix(training["OGrade"], regr.predict(X))
rfcConfusionMatrix = confusion_matrix(training["OGrade"], rfc.predict(X))
