# -*- coding: utf-8 -*-
"""
Numerical experiments on Census dataset concerning incomes.
https://www.census.gov/census2000/PUMS5.html
https://github.com/chocjy/randomized-quantile-regression-solvers/tree/master/matlab/data

Created on Tue May 17 13:53:25 2016

@author: maxime sangnier
"""


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat


def load_data(filename='census_data.mat', test_size=0.33):
    data = loadmat(filename)
    X = data.get('A')  # Educational Attainment
    X = np.asarray(X, dtype='float64')
    # Variables
#    Sex
#    Age in 30 40
#    Age in 40 50
#    Age in 50 60
#    Age in 60 70
#    Age gte 70
#    Non white
#    Unmarried
#    Education
#    Education code squared
    # Education
    #00 Not in universe (Under 3 years)
    #01 No schooling completed
    #02 Nursery school to 4th grade
    #03 5th grade or 6th grade
    #04 7th grade or 8th grade
    #05 9th grade
    #06 10th grade
    #07 11th grade
    #08 12th grade, no diploma
    #09 High school graduate
    #10 Some college, but less than 1 year
    #11 One or more years of college, no degree
    #12 Associate degree
    #13 Bachelor’s degree
    #14 Master’s degree
    #15 Professional degree
    #16 Doctorate degree
    y = data.get('b')[:, 0].reshape(-1, 1)
    del data
    ind = np.nonzero(y == 0.)[0]  # remove samples with no income
    X = np.delete(X, ind, axis=0)
    y = np.delete(y, ind, axis=0)
    X = np.delete(X, 1, axis=1)  # this variable is just ones everywhere

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                test_size=test_size, random_state=0)
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    X_test = scaler_x.transform(X_test)
    y_test = scaler_y.transform(y_test)

    return X_train, X_test, y_train, y_test
