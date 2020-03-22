# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:53:12 2020

@author: anguera
"""
import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\anguera\Desktop\Anaconda\Excel files\Assigment_2_3_e197406_prob.csv')
print(data.head())

data_train = pd.read_csv("real_train.csv")
data_test = pd.read_csv("real_test1.csv")

# Missing values check 
data_train.isnull().sum()

## Split of the train set into X and Y variables
X = data_train.drop(["label","visitTime","hour","purchaseTime"], axis=1)
Y = data_train["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.33, stratify=Y)

## Counting 'label' values
import matplotlib.pyplot as plt
data_train['label'].value_counts().plot.bar()

## We have an imbalanced train srt
data_train['label'].value_counts()


## Balancing the data set with SMOTE
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
sm = SMOTE(random_state=42, sampling_strategy=0.3)
X_sm, Y_sm = sm.fit_resample(X_train,Y_train)
rs = RandomUnderSampler(random_state =42, sampling_strategy=0.7)
X_2, Y_2 = rs.fit_resample(X_sm, Y_sm)
print('Resampled dataset shape %s' % Counter(Y_2))



## Import and use of logistic regresison model
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_2, Y_2)
predlr = logr.predict(X_test)


## Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test, predlr))


## Confusion matrix for logistic regression
from sklearn.metrics import confusion_matrix
confusion_matrix(predlr,Y_test)


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
proba_lr = logr.predict_proba(X_test)
predic_lr = proba_lr[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, predic_lr)
roc_auc = metrics.auc(fpr, tpr)



import matplotlib.pyplot as plt
plt.title('ROC of logistic regression')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


## Fit the model to test set
data_test



Xtes = data_test.drop(["label","visitTime","hour","purchaseTime"], axis=1)
Ytes = data_test["label"]


## Predict probabilities
predtes = logr.predict_proba(Xtes)
predtes


predictions = pd.DataFrame(predtes)
fin_pred = predictions.rename(columns = {0:"No_Purchase",1:"Purchase"})
fin_pred

## Merge of the two datasets
concate = pd.concat([Xtes, fin_pred], axis=1, join='inner')
finalcsv = concate[["id","Purchase"]]
finalcsv


#export it
finalcsv.to_csv('C:\Users\anguera\Desktop\Anaconda\Excel files\Assigment_2_3_e197406_prob.csv')