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


def readFoodRecData():
    print("Attempting to read data from foodRecSys package...")
    df = data_utils.get_full_core_df()
    df.head()

def getInputs():
    global filename
    while True:
        pass
        try:
            filename = str(input('Enter dataset filepath.\n'))
            break
        except:
            print("That's not a valid option!\n")
            sys.stderr.close()

getInputs()

#import Epicurious (tags) dataset
#recipes =pd.read_csv("/Users/josephmatan/downloads/epi2/epi_r.csv").dropna()
recipes =pd.read_csv(filename).dropna()

mean_rating = recipes['rating'].mean()
#recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)
recipes['target'] = np.where(recipes['rating']>= 4, 1, 0)  # TODO: decide on what is a 'positive' rating
#print(mean_rating)
#print(recipes.head())

recipe_tags = recipes.drop(['title', 'rating', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
#print(recipe_tags.head())
recipe_tags.target = recipe_tags['target']
recipe_tags.data = recipe_tags.drop(['target'], axis = 1)
#print(recipe_tags.target.head())
#print(recipe_tags.data.head())

x_train, x_test, y_train, y_test = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.30, random_state=0)

log_reg = LogisticRegression(solver = 'lbfgs')
log_reg.fit(x_train, y_train)
predictions = log_reg.predict(x_test)
#print(predictions.head())

score = log_reg.score(x_test, y_test)
print(score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)








