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
        print('=' * 100)

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
    recipes['ingredients'] = recipes['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    print('='*100)
    print('FIRST'*100)
    print (str(recipes['ingredients'].head()))
    print('='*100)


    print('='*100)
    print('VOCAB'*5)
    v = CountVectorizer(ngram_range=(2, 2))
    bigrams = v.fit_transform(recipes['ingredients'])
    vocab = v.vocabulary_
    count_values = bigrams.toarray().sum(axis=0)
    print('VOCAB'*5)
    print('='*100)

    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
        print (bg_count, bg_text)
    


    #code to vectorize everything
    





    # Not tags
    recipe_tags = recipes.drop(['title', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
    mean_rating = recipes['rating'].mean()
    # Creating new column where ratings are 1 for good and 0 for bad
    recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)

    print("MEAN RATING: "  + str(mean_rating))
    print (str(recipes.head()))
    print ("COLUMNS" + str(recipes.columns))
'''
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


if __name__ == "__main__":
    cli()
