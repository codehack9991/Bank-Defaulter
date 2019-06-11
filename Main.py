# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 00:14:54 2018

@author: pc
"""
import time

t1 = time.time()

import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold as KFold

from sklearn.ensemble import BaggingClassifier as BRC
from sklearn.ensemble import ExtraTreesClassifier as ETC

#using for loop to run the code 200 times and generate 200 different results
for z in range(1,601):
    print(z)
    #Reading features from training dataset into a pandas dataframe
    f = r'F:/Analyticity2018/train.csv'    #reading address of file
    df = pd.read_csv(f) #creating pandas dataframe

    #storing value of features from training dataset into numpy arrays
    LOAN_ID = np.array(df['LOAN_ID'])
    AMOUNT = np.array(df['AMOUNT'])
    DUE_MORTGAGE = np.array(df['DUE_MORTGAGE'])
    VALUE = np.array(df['VALUE'])
    REASON = np.array(df['REASON'])
    OCC = np.array(df['OCC'])
    TJOB = np.array(df['TJOB'])
    DCL = np.array(df['DCL'])
    CLT = np.array(df['CLT'])
    CL_COUNT = np.array(df['CL_COUNT'])
    RATIO = np.array(df['RATIO'])
    CONVICTED = np.array(df['CONVICTED'])
    VAR_1 = np.array(df['VAR_1'])
    VAR_2 = np.array(df['VAR_2'])
    VAR_3 = np.array(df['VAR_3'])

    #label of training dataset
    DEFAULTER = np.array(df['DEFAULTER'])

    #stacking features together in a matrix
    X = np.column_stack((AMOUNT, VAR_1, VAR_2, DUE_MORTGAGE, VALUE, DCL, REASON, OCC, TJOB, CL_COUNT, CL_COUNT, RATIO))

    #setting Y as the label
    Y = DEFAULTER

    #using the Imputer() function to fill in the missing values using strategy='mean'
    imputer = Imputer(copy=False)
    transformed_X = imputer.fit_transform(X)

    #fitting the model with training dataset. Model is a BaggingClassifier, with ExtraTreesClassifier as it's estimator
    model = BRC(base_estimator=ETC(n_estimators=30), n_estimators=100,bootstrap_features=True,oob_score=True,max_features = 7)
    model.fit(transformed_X,Y)

    
    '''
    #crossvalidating the model using RepeatedStratifiedKFold
    model = BRC(base_estimator=ETC(n_estimators=30), n_estimators=100,bootstrap_features=True,oob_score=True,max_features = 7)
    kfold = KFold()
    result = cross_val_score(model, transformed_X, Y, cv=kfold, scoring = 'roc_auc')
    print(result.mean())
    '''


    f = r'F:/Analyticity2018/test.csv'    #reading address of file
    df = pd.read_csv(f) #creating pandas dataframe

    #storing value of features from testing dataset into numpy arrays
    TEST_ID = np.array(df['TEST_ID'])
    LOAN_ID_T = np.array(df['LOAN_ID'])
    AMOUNT_T = np.array(df['AMOUNT'])
    DUE_MORTGAGE_T = np.array(df['DUE_MORTGAGE'])
    VALUE_T = np.array(df['VALUE'])
    REASON_T = np.array(df['REASON'])
    OCC_T = np.array(df['OCC'])
    TJOB_T = np.array(df['TJOB'])
    DCL_T = np.array(df['DCL'])
    CLT_T = np.array(df['CLT'])
    CL_COUNT_T = np.array(df['CL_COUNT'])
    RATIO_T = np.array(df['RATIO'])
    CONVICTED_T = np.array(df['CONVICTED'])
    VAR_1_T = np.array(df['VAR_1'])
    VAR_2_T = np.array(df['VAR_2'])
    VAR_3_T = np.array(df['VAR_3'])


    #stacking testing features together in a matrix
    X_T = np.column_stack((AMOUNT_T, VAR_1_T, VAR_2_T, DUE_MORTGAGE_T, VALUE_T, DCL_T, REASON_T, OCC_T, TJOB_T, CL_COUNT_T, CL_COUNT_T, RATIO_T))

    #using the Imputer() function to fill in the missing values using strategy='mean'
    imputer = Imputer(copy=False)
    transformed_X_T = imputer.fit_transform(X_T)


    #predicting the proba of final values
    PROBA = model.predict_proba(transformed_X_T)
    if z==1:
        PREDICTED = PROBA[:,1]
    else:
        PREDICTED = PREDICTED+PROBA[:,1]

#dividing by 600 as before we had added prob. 600 times
PREDICTED = PREDICTED/600

Final = pd.DataFrame({"LOAN_ID":LOAN_ID_T,"DEFAULTER":PREDICTED})
Final.to_csv("RESULT.csv", index=False) 

print((time.time()-t1)/60)











