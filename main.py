##BELLA ADDED for tensorflow model 
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestRegressor


import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import mean_squared_error


########################


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
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression

import json
import re

# Custom defined
# from models import *
#from utils import data_utils, test_utils

# New
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer




@click.group()
def cli():
    pass

#This just here for ref on how to use cli.command()... run the useage line
#and it runs the function for quick prototyping the parts
@cli.command()
def dummy():
    """
    Usage: `python main.py dummy`
    """
    raise NotImplementedError(
        "dont actually run this"
    )

@cli.command()
@click.option('--version', default="train_rating", help="csv to read")
def readFoodRecData(version):
    print("Attempting to read data from foodRecSys package...")
    df = data_utils.get_full_core_df(['recipe_id','user_id','rating'], version)
    print(df.head())


# @cli.command()
def get_json_recipes():
    print("parsing out the recipes...")
    with open('data/epicurious/full_format_recipes.json', 'r') as f:
        recipe_ingredients_dict = json.load(f)

    for i in range(10):
        print(recipe_ingredients_dict[i]['ingredients'][0])
        print('=' * 80)

    return recipe_ingredients_dict


def removeNums(x):
    # sprint x
    result = [re.sub('[^a-zA-Z ]+', '', val) for val in x]
    return result
    # print x
    # quit()
@cli.command()
def vectorize():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # recipes['ingredients'] = recipes['ingredients'].str.replace('\d+', '')
    recipes = recipes.dropna(subset=['rating', 'ingredients'])
    recipes['ingredients'] = recipes['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    ingredient_string = recipes['ingredients'].apply(lambda x : " ".join(str(word) for word in x))

    print('='*80)
    # get most common bigrams
    # Now this also takes out stop words
    sw = stopwords.words('english')
    sw.append('andor')

    v = CountVectorizer(ngram_range=(1, 2), stop_words = sw)
    bigrams = v.fit_transform(ingredient_string)
    vocab = v.vocabulary_
    count_values = bigrams.toarray().sum(axis=0)
    print('='*80)
    # print most common bigrams
    len_two = 0
    i = 0
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
        if i < 5000:
            print (bg_count, bg_text)
        i += 1
    print('\n This many words occur twice or more:')
    print(len_two)

    # Make feature vectors out of bigrams

    '''
    # Not tags
    recipe_tags = recipes.drop(['title', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
    mean_rating = recipes['rating'].mean()
    # Creating new column where ratings are 1 for good and 0 for bad
    recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)

    print("MEAN RATING: "  + str(mean_rating))
    print (str(recipes.head()))
    print ("COLUMNS" + str(recipes.columns))

    #FOR JOSEPH
    ingred_to_rating = pd.concat((recipes['ingredients'], recipes['rating'], recipes['target']), axis=1, keys=['ingredients', 'rating', 'target'])
    print ('=' * 100)
    print ("dataframe for joseph")
    print ('=' * 100)
    print (str(ingred_to_rating.head()))
    print ('=' * 100)
    print ('=' * 100)

    # Create data and target, split into train and test, I think we should sample evenly
    recipe_tags.target = recipe_tags['rating']
    recipe_tags.target = recipe_tags.target.round(0) #MAYBE CHANGE THIS
    recipe_tags.target = pd.to_numeric(recipe_tags.target, downcast='integer')
    #print(recipe_tags.target.head())
    print(recipe_tags.target.value_counts())
    recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.3, random_state=42)
'''
    
@cli.command()
def LinearRegression():
    #import Epicurious (tags) dataset
    recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna(subset=['rating'])

    # Not tags
    recipe_tags = recipes.drop(['title'], axis = 1)
    # Create data and target, split into train and test, I think we should sample evenly
    recipe_tags.target = recipe_tags['rating']
    recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)

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
    print('Confusion Matrix')
    print(metrics.confusion_matrix(y_test, predictions))
    print('linear coef: ' + str(lin.coef_))
    print(lin.coef_)

#import Epicurious (tags) dataset
recipes = pd.read_csv("data/epicurious/epi_r.csv").dropna()

# Doc2Vec implementation
@cli.command()
def doc2vec():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # Not tags
    recipe_tags = recipes.drop(['title', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
    mean_rating = recipes['rating'].mean()
    # Creating new column where ratings are 1 for good and 0 for bad
    recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)
    ingred_to_rating = pd.concat((recipes['ingredients'], recipes['rating'], recipes['target']), axis=1, keys=['ingredients', 'rating', 'target'])
    print(ingred_to_rating.shape)
    #ingred_to_rating['ingredients'] = ingred_to_rating.ingredients.apply(lambda x: ' '.join(str(x)))


    print(ingred_to_rating['ingredients'].apply(lambda x: len(x.split(' '))).sum())
    print(ingred_to_rating.head())
    ingred_to_rating['ingredients'] = ingred_to_rating['ingredients'].apply(cleanText)
    train, test = train_test_split(ingred_to_rating, test_size=0.3, random_state=42)
    train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['ingredients']), tags=[r.rating]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['ingredients']), tags=[r.rating]), axis=1)
        # Counter = Counter(split_it)
    # most_occur = Counter.most_common(2000) # TODO: change number
    # most_common_words = [word for word, word_count in most_occur]
    # with open('dict.csv', 'w') as f:
    #     for item in most_common_words:
    #         f.write("%s\n" % (str(item)



def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text




def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('rating')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

# neuralnet implementation riperonies.
@cli.command()
def neuralnet():
    #Importing Libraries for data preparation

    # #Read Necessary files
    # recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna(subset=['rating'])
    # train, test = train_test_split(recipes, test_size=0.2)
    # # train, val = train_test_split(train, test_size=0.2)
    # #Combined both Train and Test Data set to do preprocessing together 
    # # and set flag for both as well
    # train['Type'] = 'Train' 
    # test['Type'] = 'Test'
    # fullData = pd.concat([train,test],axis=0)


    # #Identifying ID, Categorical
    # flag_col= ['Type']
    # target_col = ["rating"]
    # cat_cols= list(recipes.columns)[:]
    # cat_cols.remove('rating')
    # cat_cols.remove('title')
    # num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(target_col)-set(flag_col))
    # # Combined numerical and Categorical variables
    # num_cat_cols = num_cols+cat_cols
    # #Create a new variable for each variable having missing value with VariableName_NA 
    # # and flag missing value with 1 and other with 0
    # for var in num_cat_cols:
    #     if fullData[var].isnull().any()==True:
    #         fullData[var+'_NA']=fullData[var].isnull()*1
    # #Impute numerical missing values with mean
    # fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean())
    # #Impute categorical missing values with -9999
    # fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)


    # #create label encoders for categorical features
    # for var in cat_cols:
    #     number = LabelEncoder()
    #     fullData[var] = number.fit_transform(fullData[var].astype('str'))

    # #normalize independent variables between 0 and 1 oto converge faster
    # #maybe undo
    # # features = list(set(list(fullData.columns))-set(target_col))
    # # fullData[features] = fullData[features]/fullData[features].max()

    # #make validation set 
    # print(fullData)
    # train=fullData[fullData['Type']==1]
    # test=fullData[fullData['Type']==0]
    # features=list(set(list(fullData.columns))-set(target_col)-set(flag_col))
    # # print(features)
    # X = train[features].values
    # print("THIS IS X: ", X)
    # y = train[target_col].values
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=42)

    # print(X_train[0])

    #import Epicurious (tags) dataset
    recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna()

    # Not tags
    recipe_tags = recipes.drop(['title'], axis = 1)
    # Create data and target, split into train and test, I think we should sample evenly
    recipe_tags.target = recipe_tags['rating']
    recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)

    # #print(recipe_tags.target.head())
    # print(recipe_tags.target.value_counts())
    x_train, x_test, y_train, y_test = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.3, random_state=42)








    random.seed(42)
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(x_train, y_train)
    print("fit to random forest")

    # Define model
    model = Sequential()
    model.add(Dense(100, input_dim=678, activation= "relu"))
    model.add(Dense(50, activation= "relu"))
    model.add(Dense(1))
    model.summary() #Print model Summary

    # Compile model
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    
    # Fit Model
    model.fit(x_train, y_train, epochs=10)

    pred= model.predict(x_valid)
    score = np.sqrt(mean_squared_error(y_valid,pred))
    print ("neural network", score)



    #Prediction using Random Forest 
    y_valid_rf = rf.predict(x_valid)
    score = np.sqrt(mean_squared_error(y_valid_rf,pred))
    print ("random forest", score)









    #ATTEMPT 1: 
    # recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna(subset=['rating'])
    # train, test = train_test_split(recipes, test_size=0.2)
    # train, val = train_test_split(train, test_size=0.2)
    # print(len(train), 'train examples')
    # print(len(val), 'validation examples')
    # print(len(test), 'test examples')

    # #change to tf.data dataset from panda df
    # batch_size = 5 # change this to 32 idk what it does though
    # train_ds = df_to_dataset(train, batch_size=batch_size)
    # val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    # test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # # this is how you would iterate through a batch
    # # for feature_batch, label_batch in train_ds.take(1):
    # #   print('Every feature:', list(feature_batch.keys()))
    # #   print('A batch of ages:', feature_batch['age'])
    # #   print('A batch of targets:', label_batch )
    # #
    # # and this is what output would look like:
    # # Every feature: ['restecg', 'exang', 'ca', 'fbs', 'sex', 'oldpeak', 'chol', 'thalach', 'thal', 'cp', 'slope', 'age', 'trestbps'] 
    # # A batch of ages: tf.Tensor([60 37 46 57 57], shape=(5,), dtype=int32)
    # # A batch of targets: tf.Tensor([1 0 0 1 0], shape=(5,), dtype=int32)

    # calories = feature_column.numeric_column("calories")
    # demo(calories)













































if __name__ == "__main__":
    cli()
