import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pickle
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from imblearn.over_sampling import RandomOverSampler

def clean_churn_df(df):
    df.columns = df.columns.str.strip()
    df = pd.concat([df, pd.get_dummies(df['Departments'])], axis=1)
    df = pd.concat([df, pd.get_dummies(df['salary'])], axis=1)
    df.drop(['left','Departments','salary'],axis=1,inplace=True)
    return df

def score_model(model, X_train, y_train):
    cv_results = cross_validate(model, X_train, y_train, cv=5, return_train_score=False)
    return np.mean(cv_results['test_score'])

def model_baseline(X_train, y_train):
    #lm1 = LogisticRegression()     #1 feature logistic regression
    #lm1_score = score_model(lm1, X_train['satisfaction_level'], y_train)

    lm2 = LogisticRegression() #all features
    lm2_score = score_model(lm2, X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn_score = score_model(knn, X_train, y_train)

    gnb = GaussianNB()
    gnb_score = score_model(gnb, X_train, y_train)

    mnb = MultinomialNB()
    mnb_score = score_model(mnb, X_train, y_train)

    svm_model = svm.SVC()
    svm_score = score_model(svm_model, X_train, y_train)

    rf = RandomForestClassifier()
    rf_score = score_model(rf, X_train, y_train)

    #maybe try XGB for fun?
    #clf = XGBClassifier().fit(X_train, y_train)

    return lm2_score, knn_score, gnb_score, mnb_score, svm_score, rf_score

def RMSE(validation_points, prediction_points):
   """
   Calculate RMSE between two vectors
   """
   x = np.array(validation_points)
   y = np.array(prediction_points)

   return np.sqrt(np.mean((y-x)**2))
