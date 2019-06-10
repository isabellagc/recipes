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
from nltk.util import ngrams
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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

from main import get_accuracy

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

@cli.command()
@click.option('--vals', default=100, help='how many of top words to include') #how manhy of the top 5000 words to take ink
@click.option('--tagnum', default = 675, help='how many of top tags to include')
@click.option('--notags', default=False, is_flag=True, help='whether to zip up with the tags')
def finalDF(vals, tagnum, notags):
     #real df with tags 
    recipes = pd.read_csv('data/epicurious/epi_r.csv')



    recipes = recipes.drop(axis=1, columns=['title'])
    just_tags = recipes.drop(axis=1, columns=['calories','sodium','fat','protein','rating'])
    ##GET THE MOST COMMON TAGS DELETE THE ONES THAT ARE SHITE
    summies = just_tags.sum(axis = 0, skipna = True) 
    summies.sort_values(ascending=False, inplace=True)

    
    # dont_use = []
    # threshold = 400
    # total = 0
    # for title, val in summies.iteritems():
    #     if val < threshold:
    #         dont_use.append(title)
    # print(len(dont_use))
    # print(dont_use)
    summies =  summies.head(tagnum)
    best = summies.index
    print("best"*20)
    print(best)
    print('all'*20)
    print(just_tags.columns)

    diff = list(set(just_tags.columns) - set(best))
    print('diff'*20)
    print(diff)
    

    recipes = recipes.drop(axis=1, columns=diff)
    # print(summies)
    print(recipes)
    print(recipes.shape)
    
   
    #just the ingredient part 
    json_recipes = get_json_recipes()
    ingredients = pd.DataFrame.from_dict(json_recipes, orient='columns')
    ingredients = ingredients.dropna(subset=['rating', 'ingredients'])
    ingredients['full_ingredients'] = ingredients['ingredients']
    ingredients['ingredients'] = ingredients['ingredients'].apply(removeNums)
    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    ingredient_string = ingredients['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    

    print('='*80)
    # get most common bigrams
    # Now this also takes out stop words
    sw = stopwords.words('english')
    sw.append('andor')
    units_list = ['cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'ounce', 'ounces', 'pound', 'pounds', 'lb', 'lbs','small', 'large', 'inch' ]
    sw += units_list

    v = CountVectorizer(ngram_range=(1, 2), stop_words = sw)
    bigrams = v.fit_transform(ingredient_string)
    vocab = v.vocabulary_
    count_values = bigrams.toarray().sum(axis=0)
    print('='*80)
    # Make list of 5000 most common words and bigrams
    i = 0
    most_common_words = []

    print("entering most common words, going to pick the top " + str(int(vals)))
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:int(vals)]:
        print (bg_count, bg_text)
        most_common_words.append(bg_text.strip())
    print('\nVector of most common words and bigrams is this long: ' + str(len(most_common_words)))
    most_common_words_set = set(most_common_words)

    print("now making the feature vector per list: ")
    # Makes a feature vector for each list of ingredients
    feature_vectors = []

    for i, row in tqdm(ingredients.iterrows()):
        feature_vec = []
        feature_vec_old = []
        ingred_list = list(row['full_ingredients'])  
        # print(ingred_list)
        ingred_string = ' '.join(ingred_list)
        # print('this is the string: ')
        # print(ingred_string)

        # quit()
        ingred = ingred_string.split()
        # print(ingred) 
        for word in most_common_words:
            used = False
            # print('LOOKING FOR : ' + word)
            if word in ingred_string:
                feature_vec_old.append(1)
                # print("entered here with word : " + word) #it is somewhere in this list of ingredients
                for line in ingred_list: #each instruction
                    if used:
                        break
                    # print('checking the line : ' + line + ' for word ' + word)
                    if word in line: #if its in this line
                        # print('found the word in this line : ' + line)
                        line = line.lower()
                        no_num = re.sub('[^a-zA-Z ]+', '', line) 
                        tokens = [token for token in no_num.split(" ") if token != ""]
                        bigrams = list(ngrams(tokens, 2))
                        bigrams = [' '.join(x) for x in bigrams]
                        line_grams = tokens + bigrams

                        # print('going to check fit against all these: ')
                        # print(line_grams)
                        for token in line_grams: #each word in that instruction
                            if used:
                                break
                            # print('checking if token ' + token + " is equal to " + word) 
                            if token == word:
                                used = True
                                num = re.search('([0-9]+[,.]?[0-9]*([\/][0-9]+[,.]?[0-9]*)*)', line)
                                if num:
                                    try:
                                        num = float(Fraction(num.group(1)))
                                    except:
                                        print('got em... ' + num.group(1) + ' with ingrs : ' + line )
                                        num = 1
                                    # print('found a number, ' + str(num) + ' corresponding to : ' + word)
                                    feature_vec.append(num)
                                else:
                                    # print('no number found in this row to associate with : ' + word)
                                    feature_vec.append(1)
                if not used:
                    # print('weird thing where we didnt end up findn...')
                    # print(word)
                    feature_vec.append(1)

                            
                        

            else:
                feature_vec.append(0)
                feature_vec_old.append(0)

            if len(feature_vec) != len(feature_vec_old):
                print("new len: " + str(len(feature_vec)) + " old len " + str(len(feature_vec_old)))
                print('this happened on this word: ' + word)
                print(ingred_string)
                print(ingred_list)
                quit()
        # print('the feature vec for this row is now: ' + str(feature_vec))

        
        # feature_vec_old = []
        # for word in most_common_words:
        #     if word in ingred_string:
        #         # print("OLD entered here with word : " + word)
        #         feature_vec_old.append(1)
        #     else:
        #         feature_vec_old.append(0)
        # print('this was the old feature  vec: ' + str(feature_vec_old))
        # # quit()


        if len(feature_vec) != len(most_common_words):
            print("length was actually " + str(len(feature_vec))) 
            print(most_common_words)
            print(feature_vec)
            feature_vec_old = []
            for word in most_common_words:
                if word in ingred_string:
                    # print("OLD entered here with word : " + word)
                    feature_vec_old.append(1)
                else:
                    feature_vec_old.append(0)
            print('this was the old feature  vec: ' + str(feature_vec_old))
            print(ingred_list)
            quit()
        feature_vectors.append(feature_vec)
    print("done with the feature vectors ")
    for vec in feature_vectors:
        print(vec)

    ingr =  pd.DataFrame(feature_vectors, columns = most_common_words)
    print("this is the shape of the ingredient matrix: " + str(ingr.shape))
    print(ingr.head())
    
    if notags:
        ingr['rating'] = recipes['rating']
        combined_df = ingr
    else:
        combined_df = pd.concat([recipes, ingr], axis=1)
        # combined_df = combined_df.drop(['title'], axis = 1)     
        # Remove zero ratings
        combined_df = combined_df[combined_df.rating > 1]
        combined_df = combined_df[combined_df.rating.notnull()]


  
    print("this is the new matrix shape: " + str(combined_df.shape))
    final = combined_df.dropna()
    print(final.head())
    print("this is the final matrix shape without null: " + str(final.shape))
    final.to_pickle('final_dataframe_linreg.pkl')



#recipe_tags = recipes.drop(['title','protein','calories','fat','sodium'], axis = 1)
#recipe_tags = recipes.drop(['title'], axis = 1)
@cli.command()
@click.option('--vals', default=100, help='how many of top words to include') #how manhy of the top 5000 words to take ink
def finalDF_OLD(vals):
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
    units_list = ['cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'ounce', 'ounces', 'pound', 'pounds', 'lb', 'lbs']
    sw += units_list

    v = CountVectorizer(ngram_range=(1, 2), stop_words = sw)
    bigrams = v.fit_transform(ingredient_string)
    vocab = v.vocabulary_
    count_values = bigrams.toarray().sum(axis=0)
    print('='*80)
    # Make list of 5000 most common words and bigrams
    i = 0
    most_common_words = []
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:int(vals)]:
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

    combined_df = combined_df[combined_df.rating > 1]
    combined_df = combined_df[combined_df.rating.notnull()]
    
    print(final.head())
    print("this is the final matrix shape without null: " + str(final.shape))
    final.to_pickle('final_dataframe_linreg.pkl')

@cli.command()
def linreg():
    recipes = pd.read_pickle('final_dataframe.pkl')
    
    # Create data and target, split into train and test, I think we should sample evenly
    recipes.target = recipes['rating']
    #recipe_tags.data = recipe_tags.drop(['rating'], axis = 1)
    recipes.data = recipes.drop(['rating'], axis = 1)
    #print recipe_tags.data
    # quit()

    # #print(recipe_tags.target.head())
    # print(recipe_tags.target.value_counts())
    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)

    # Run logistic regression, print results
    lin = LinearRegression()
    lin.fit(x_train, y_train)
    name_and_coef = {}
    for idx, col_name in enumerate(x_train.columns):
        name_and_coef[lin.coef_[idx]] = col_name
        #print("The coefficient for {} is {}".format(col_name, lin.coef_[idx]))
    sorted_coef = sorted(lin.coef_, reverse = True)
    for coef in sorted_coef:
        print ("The coefficient for {} is {}".format(name_and_coef[coef],coef))
    
    print ('Linear Regression:')
    predictions = lin.predict(x_test)
    print ('Score:')
    print (lin.score(x_test, y_test),'\n')

    print (mean(y_train.values))

    print (y_test.values)
    print (mean(y_test.values))
    index = []
    for i in range(0,predictions.shape[0]):
        if abs(predictions[i]) > 100:
            index.append(i)
            #print(predictions[i])
            #print("first")
    new_predictions = np.delete(predictions, index)
    new_y_test = np.delete(y_test.values, index)
    print (len(index))
    print (index)
    print (new_predictions)
    print (mean(new_predictions))

    print ("MEAN ABSOLUTE ERROR : " + str(mean_absolute_error(new_y_test, new_predictions)))
    print ("ROOT MEAN SQUARED ERROR : " + str(np.sqrt(mean_squared_error(new_y_test, new_predictions))))
    print ("R2 : " + str(r2_score(new_y_test, new_predictions)))
    get_accuracy(new_predictions, new_y_test)
    #diffs = pd.DataFrame({'Actual': y_test, 'Predicted': new_predictions})  


    '''
    lasso = Lasso(max_iter = 1000)
    parameters = {'alpha': [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(lasso, parameters, cv=3, verbose = 100, n_jobs = 3)
    grid_search.fit(x_train, y_train)
    print('params')
    print(grid_search.best_params_)

    lasso = Lasso(max_iter = 100000, alpha = 10)
    lasso.fit(x_train, y_train)
    coeff_used = np.sum(lasso.coef_!=0)
    print('Coefficient count: ')
    print(len(lin.coef_))
    print('After LASSO: ')
    print(coeff_used)
    print('score:')
    print(lasso.score(x_test, y_test))
    print()
    '''




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



















