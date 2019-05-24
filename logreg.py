import click
import torch
import time
import pandas as pd
import random
from tqdm import tqdm

#-----brought in from example project might delete later
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure


# Custom defined
# from models import *
from utils import data_utils, test_utils

# NEW

import scipy
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#import Epicurious (tags) dataset
recipes = pd.read_csv('./data/epicurious/epi_r.csv').dropna()

# Not tags
recipe_tags = recipes.drop(['title', 'calories', 'protein', 'fat', 'sodium'], axis = 1)

# Create data and target, split into train and test, I think we should sample evenly
recipe_tags.target = recipe_tags['rating']
recipe_tags.target = recipe_tags.target.round(0)
recipe_tags.target = pd.to_numeric(recipe_tags.target, downcast='integer')
#print(recipe_tags.target.head())
print(recipe_tags.target.value_counts())
recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.3, random_state=42)

# Run logistic regression, print results
mult_log_reg = LogisticRegression(multi_class = 'auto', solver = 'liblinear', max_iter = 1000)
mult_log_reg.fit(x_train, y_train)
print('multiple logistic regression:')
predictions = mult_log_reg.predict(x_test)
print('Score:')
print(mult_log_reg.score(x_test, y_test),'\n')
print('Confusion Matrix')
print(metrics.confusion_matrix(y_test, predictions))









