import click
import time
import pandas as pd
import random
from tqdm import tqdm
import math

#-----brought in from example project might delete later
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statistics import mean

# Custom defined
# from models import *
from utils import data_utils, test_utils

# NEW

import scipy
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#import Epicurious (tags) dataset
recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna()

# Not tags
recipe_tags = recipes.drop(['title'], axis = 1)
# Create data and target, split into train and test, I think we should sample evenly
recipe_tags.target = recipe_tags['rating']
recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)
print recipe_tags.data
# quit()

# #print(recipe_tags.target.head())
# print(recipe_tags.target.value_counts())
x_train, x_test, y_train, y_test = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.3, random_state=42)

# Run logistic regression, print results
lin = LinearRegression()
lin.fit(x_train, y_train)
print('multiple logistic regression:')
predictions = lin.predict(x_test)
print('Score:')
print(lin.score(x_test, y_test),'\n')

print mean(y_train.values)

print y_test.values
print mean(y_test.values)
print predictions
print mean(predictions)

print "MEAN ABSOLUTE ERROR : " + str(mean_absolute_error(y_test.values, predictions))
print "MEAN SQUARED ERROR : " + str(np.sqrt(mean_squared_error(y_test.values, predictions)))
diffs = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})  

lower = 0
upper = 0


    	

print "upper " + str(upper)
print "lower " + str(lower)


# print('linear coef: ' + str(lin.coef_))
print(len(lin.coef_))

cols = y_train.columns

zipped = dict(zip(cols, lin.coef_))
# print(sorted(zipped.items(), key = lambda kv: kv[1])) 



# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# numys = recipes.select_dtypes(include=numerics)
# vals = numys.sum()
# print numys.sum()
# count = 0
# for val in vals:
# 	if val < 40:
# 		count += 1

# numyframe = numys.to_frame()
# print numyframe.head()










