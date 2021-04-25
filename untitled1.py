# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:43:59 2020

@author: PC
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Impot the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

'''from sklearn.cross_validation import train_test_split
X_train,X_test,Y_test,Y_train = train_test_split(X,Y,test_size = 0.2, random_state = 0 )'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)


#Visualising the Linear Regression results
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
lin_reg.predict(6.5)
lin_reg_2.predict(poly_reg.fit_transform(6.5))
