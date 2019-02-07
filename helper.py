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
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring='recall')
    return np.mean(cv_results['test_score'])

def score_model_no_cv(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    return recall_score(model.predict(X_val), y_val)


def model_baseline_no_cv(X_train, y_train, X_val, y_val):
    lm2 = LogisticRegression() #all features
    lm2_score = score_model_no_cv(lm2, X_train, y_train, X_val, y_val)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn_score = score_model_no_cv(knn, X_train, y_train, X_val, y_val)

    gnb = GaussianNB()
    gnb_score = score_model_no_cv(gnb, X_train, y_train, X_val, y_val)

    mnb = MultinomialNB()
    mnb_score = score_model_no_cv(mnb, X_train, y_train, X_val, y_val)

    svm_model = svm.SVC()
    svm_score = score_model_no_cv(svm_model, X_train, y_train, X_val, y_val)

    rf = RandomForestClassifier()
    rf_score = score_model_no_cv(rf, X_train, y_train, X_val, y_val)

    #maybe try XGB for fun?
    #clf = XGBClassifier().fit(X_train, y_train)

    return lm2_score, knn_score, gnb_score, mnb_score, svm_score, rf_score

def model_baseline(X_train, y_train):
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

def split_with_dupe_rows_in_train(churn_df):
    #find dupes
    churn_df_dupes = churn_df[churn_df.duplicated()==True]
    rest_df = churn_df[churn_df.duplicated()==False]

    #randomly split the non dupes
    X_train, X_holdout, y_train, y_holdout = train_test_split(rest_df, rest_df['left'], \
                                                            test_size=0.2, random_state=41)
    #set aside a validation set (approx. 20%)
    X_val = X_train.iloc[:2000,:]
    y_val = y_train.iloc[:2000]
    X_train = X_train.iloc[2000:,:]
    y_train = y_train.iloc[2000:]

    #add dupes into training data
    y_dupes = churn_df_dupes['left']
    y_train = y_train.append(y_dupes)
    X_train = pd.concat([X_train, churn_df_dupes])

    #clean your data
    X_train = clean_churn_df(X_train)
    X_val = clean_churn_df(X_val)
    X_holdout = clean_churn_df(X_holdout)

    return X_train, X_val, X_holdout, y_train, y_val, y_holdout

def rf_no_cv_iterx(X_train, y_train, X_val, y_val, iters):
    scores = []
    for i in range(iters):
        rf = RandomForestClassifier()
        rf_score = score_model_no_cv(rf, X_train, y_train, X_val, y_val)
        scores.append(rf_score)
    return np.array(scores).mean()

def RMSE(validation_points, prediction_points):
   """
   Calculate RMSE between two vectors
   """
   x = np.array(validation_points)
   y = np.array(prediction_points)

   return np.sqrt(np.mean((y-x)**2))
