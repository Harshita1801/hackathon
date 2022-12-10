# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:47:38 2022

TEAM MEMBERS :
    Harshitha S
    Harshita R
    K Likhitha
    Hamsa A
"""
import os 
import pandas as pd

os.chdir("C:/Users/harsh/OneDrive/Desktop/h datasets")
unemployment_csv=pd.read_excel('unemployment.xlsx')
crime_csv=pd.read_csv('crime.csv.xls')
infant_csv=pd.read_excel('INFANT MR.xlsx')
literacy_csv=pd.read_excel('literacy.xlsx')
malnutrition_csv=pd.read_excel('malnutrition.xlsx')
population_csv=pd.read_excel('population.xlsx')
poverty_line=pd.read_excel('poverty line.xlsx')
data=pd.read_csv('povertydata.csv.xls')

unemployment_csv.isnull().sum()
unemployment_csv.describe()

crime_csv.isnull().sum()
crime_csv.describe()

infant_csv.isnull().sum()
infant_csv.describe()

literacy_csv.isnull().sum()
literacy_csv.describe()

malnutrition_csv.isnull().sum()
malnutrition_csv.describe()

population_csv.isnull().sum()
population_csv.describe()

poverty_line.isnull().sum()
poverty_line.describe()

data.describe()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



sns.lmplot(x='unemployment',y='poverty %',data=data,fit_reg=False,hue='State/UT',legend=True,palette='Set1')
sns.lmplot(x='poverty %',y='population',data=data,fit_reg=False,hue='State/UT',legend=True,palette='Set1')
sns.lmplot(x='poverty %',y='stunting',data=data,fit_reg=False,hue='State/UT',legend=True,palette='Set1')
sns.lmplot(x='underweight',y='poverty %',data=data,fit_reg=False,hue='State/UT',legend=True,palette='Set1')
sns.lmplot(x='literacy %',y='poverty %',data=data,fit_reg=False,hue='State/UT',legend=True,palette='Set1')
sns.lmplot(x='poverty %',y='crime',data=data,fit_reg=False,hue='State/UT',legend=True,palette='Set1')


#REGRESSION 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

x2 = data.drop(['poverty %','State/UT'], axis='columns', inplace=False)
y2 = data['poverty %']



X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size=0.3, random_state = 3)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)

#Linear Regression

lgr=LinearRegression(fit_intercept=True)
model_lin2=lgr.fit(X_train1,y_train1)
pov_prediction=lgr.predict(X_test1)

lin_mse2 = mean_squared_error(y_test1,pov_prediction )
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

r2_lin_test2=model_lin2.score(X_test1,y_test1)
r2_lin_train2=model_lin2.score(X_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)


#Random Forest
rf2 = RandomForestRegressor(n_estimators = 100,max_features='auto',
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)


model_rf2=rf2.fit(X_train1,y_train1)


pov_prediction_rf2 = rf2.predict(X_test1)


rf_mse2 = mean_squared_error(y_test1, pov_prediction_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)


r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)      
