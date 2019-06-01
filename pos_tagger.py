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

from sklearn.feature_extraction.text import CountVectorizer


import os
java_path = "C:\\Program Files\\Java\\jdk1.8.0_102\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path

from nltk.tag.stanford import StanfordPOSTagger
path_to_model = "stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger"
path_to_jar = "stanford-postagger-2018-10-16/stanford-postagger.jar"
tagger=StanfordPOSTagger(path_to_model, path_to_jar)
tagger.java_options='-mx4096m'          ### Setting higher memory limit for long sentences
#sentence = 'This is testing'
#print tagger.tag(sentence.split())

def get_json_recipes():
    print("parsing out the recipes...")
    with open('data/epicurious/full_format_recipes.json', 'r') as f:
        recipe_ingredients_dict = json.load(f)

    for i in range(10):
        print(recipe_ingredients_dict[i]['ingredients'][0])
        print('=' * 100)

    return recipe_ingredients_dict

def vectorize():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    # recipes['ingredients'] = recipes['ingredients'].str.replace('\d+', '')
    recipes = recipes.dropna(subset=['rating', 'ingredients'])
    recipes['ingredients'] = recipes['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    recipes['ingredients'] = recipes['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    print(tagger.tag(recipes['ingredients'].split()))
    '''print('='*100)
    print('FIRST'*100)
    print (str(recipes['ingredients'].head()))
    print('='*100)


    print('='*100)
    print('VOCAB'*20)
    v = CountVectorizer(ngram_range=(1, 2))
    vocab = v.fit(recipes['ingredients']).vocabulary_
    print vocab
    print('VOCAB'*20)
    print('='*100)'''



    #code to vectorize everything


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
    '''
    



    
