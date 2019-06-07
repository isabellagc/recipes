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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statistics import mean

# Custom defined
# from models import *
from utils import data_utils, test_utils

# NEW

import json
import re

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

import scipy
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

@click.group()
def cli():
    pass

def get_json_recipes():
    print("parsing out the recipes...")
    with open('data/epicurious/full_format_recipes.json', 'r') as f:
        recipe_ingredients_dict = json.load(f)
    '''
    for i in range(10):
        print(recipe_ingredients_dict[i]['ingredients'][0])
        print('=' * 80)
    '''
    return recipe_ingredients_dict


def removeNums(x):
    # sprint x
    result = [re.sub('[^a-zA-Z ]+', '', val) for val in x]
    return result
    # print x
    # quit()
#@cli.command()
'''
def vectorize():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # recipes['ingredients'] = recipes['ingredients'].str.replace('\d+', '')
    recipes = recipes.dropna(subset=['rating', 'ingredients'])

    #tags = pd.read_csv('data/epicurious/epi_r.csv').dropna()
    #merged = pd.merge(recipes, tags, on='title')

    recipes['ingredients'] = recipes['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    ingredient_string = recipes['ingredients'].apply(lambda x : " ".join(str(word) for word in x))

    #print('='*80)
    # get most common bigrams
    # Now this also takes out stop words
    sw = stopwords.words('english')
    sw.append('andor')

    v = CountVectorizer(ngram_range=(1, 2), stop_words = sw)
    bigrams = v.fit_transform(ingredient_string)
    vocab = v.vocabulary_
    count_values = bigrams.toarray().sum(axis=0)
    #print('='*80)
    # Make list of 5000 most common words and bigrams
    i = 0
    most_common_words = []
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:250]:
        #print (bg_count, bg_text)
        most_common_words.append(bg_text)
    #print('\nVector of most common words and bigrams is this long: ')
    #print(len(most_common_words))

    # Makes a feature vector for each list of ingredients
    ingred_to_rating = pd.concat((recipes['ingredients'], recipes['rating']), axis=1, keys=['ingredients', 'rating'])
    feature_vectors = []
    for i, row in ingred_to_rating.iterrows():
        feature_vec = []
        ingred = str(row['ingredients'])
        for word in most_common_words:
            if word in ingred:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
        feature_vectors.append(feature_vec)
    print(len(feature_vectors))
    return feature_vectors

#import Epicurious (tags) dataset
#recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna()
json_recipes = get_json_recipes()
recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
recipe_tags = recipes.dropna(subset=['rating', 'ingredients'])
features = pd.DataFrame(np.array(vectorize()).reshape(-1,250))'''

# Not tags

#recipe_tags = recipes.drop(['title','protein','calories','fat','sodium'], axis = 1)
#recipe_tags = recipes.drop(['title'], axis = 1)
@cli.command()
def finalDF():
     #real df with tags 
    recipes = pd.read_csv('data/epicurious/epi_r.csv')

    #just the ingredient part 
    json_recipes = get_json_recipes()
    ingredients = pd.DataFrame.from_dict(json_recipes, orient='columns')
    ingredients = ingredients.dropna(subset=['rating', 'ingredients'])
    ingredients['ingredients'] = ingredients['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    ingredient_string = ingredients['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    

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
    # Make list of 5000 most common words and bigrams
    i = 0
    most_common_words = []
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:100]:
        print (bg_count, bg_text)
        most_common_words.append(bg_text)
    print('\nVector of most common words and bigrams is this long: ' + str(len(most_common_words)))


    # Makes a feature vector for each list of ingredients
    feature_vectors = []
    for i, row in ingredients.iterrows():
        feature_vec = []
        ingred = str(row['ingredients'])
        for word in most_common_words:
            if word in ingred:
                feature_vec.append(1)
            else:
                feature_vec.append(0)
        feature_vectors.append(feature_vec)

    ingr =  pd.DataFrame(feature_vectors, columns = most_common_words)
    print("this is the shape of the ingredient matrix: " + str(ingr.shape))
    print(ingr.head())
    # ingr['rating'] = recipes['rating']
    combined_df = pd.concat([recipes, ingr], axis=1)
    print("this is the new matrix shape: " + str(combined_df.shape))
    combined_df = combined_df.dropna()
    #TODO: dropping the title until we can embed it
    final = combined_df.drop(['title'], axis = 1)
    print(final.head())
    print("this is the final matrix shape without null: " + str(final.shape))
    final.to_pickle('final_dataframe_linreg.pkl')

@cli.command()
def linreg():
    recipes = pd.read_pickle('final_dataframe_linreg.pkl')
    
    # Create data and target, split into train and test, I think we should sample evenly
    recipes.target = recipes['rating']
    #recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)
    recipes.data = recipes.drop(['rating', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
    #print recipe_tags.data
    # quit()

    # #print(recipe_tags.target.head())
    # print(recipe_tags.target.value_counts())
    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)

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
    index = []
    for i in range(0,predictions.shape[0]):
        if abs(predictions[i]) > 100:
            index.append(i)
            #print(predictions[i])
            #print("first")
    new_predictions = np.delete(predictions, index)
    new_y_test = np.delete(y_test.values, index)
    print(len(index))
    print(index)
    print new_predictions
    print mean(new_predictions)

    print "MEAN ABSOLUTE ERROR : " + str(mean_absolute_error(new_y_test, new_predictions))
    print "MEAN SQUARED ERROR : " + str(np.sqrt(mean_squared_error(new_y_test, new_predictions)))
    print "R2 : " + str(r2_score(new_y_test, new_predictions))
    #diffs = pd.DataFrame({'Actual': y_test, 'Predicted': new_predictions})  



'''
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
# print numyframe.head()'''


if __name__ == "__main__":
    cli()



















