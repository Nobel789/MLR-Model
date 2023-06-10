#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 00:44:42 2023

@author: myyntiimac
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/Users/myyntiimac/Desktop/Investment.csv")
df
# We find 50 rows and 5 columns
df.isnull().any()
# after check the dtaset if there is any null value, so we didnt find the null value 
#we qre going to split the independent and dependent variable
# so our dependent variable is profit which is continous , so our prblem is regression 
#split the DV and IDV by iloc+numeric split
X=df.iloc[:, :-1]
X
y=df.iloc[:,4]
y
X=pd.get_dummies(X)

# as our independent variable list contain the catagorical variable 
# so it need to change with dummies, for this first define which column need to cnvert
#then apply get_dummies function
column_to_encode = df['State']  
encoded_column = pd.get_dummies(column_to_encode)
# Step 2: Drop the original column
df = df.drop('State', axis=1)
df
#Step 3: Concatenate the encoded column with the DataFrame
df = pd.concat([df, encoded_column], axis=1)
df
#Now we are going to split the data for training where test data will be 30%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state = 0 ) 

#now we trained the LR model with train data  by clling LR model from sklearn.lenearmodel
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
#creaet the regressor object for LineareRegression
regressor.fit(X_train, y_train)
#then chck the  performance of trained model with X_test to find y_predected
y_pred = regressor.predict(X_test)
y_pred
#Now our model is build but need to validate, for validating  we check coefficient , intercept, variance and bias score
regressor.coef_
regressor.intercept_

bias=regressor.score(X_train, y_train)
bias
variance=regressor.score(X_test, y_test)
variance
# to go main operation of OLS, we have to add constant to independent list
#if you check our multiple linerae equation we have B0 is constant but if you check our dataset we dont have any constant hear, thats why we will add  intercept as constant for everyrow
# our resulted intercept is_ 42660 or default one , lets do it by 42660
# lets append this value to 50 observation

X = np.append(arr=np.full((50, 1), 4266), values=X, axis=1) 
X
#Then do the recursive feature elemination with OLS
#lets look for predictor with highest p-value using stats model library called summary()

import statsmodels.api as sm
X_opt1 = X[:,[0,1,2,3,4,5]]

#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt1).fit()

regressor_OLS.summary()
import statsmodels.api as sm
X_opt1 = X[:,[0,1]]

#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt1).fit()

regressor_OLS.summary()
