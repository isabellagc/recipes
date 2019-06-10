##BELLA ADDED for tensorflow model 
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import math 
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestRegressor
from keras.wrappers.scikit_learn import KerasRegressor


import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier     
from sklearn import preprocessing
from keras import losses
from keras import backend as K
from fractions import Fraction
########################


import click
import torch
import time
import pandas as pd
import random
from tqdm import tqdm
from nltk.util import ngrams
#-----brought in from example project might delete later
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics


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

# Prints accuracy metrics for a list of predictions
# You need to give pandas dataframe columns corresponding
# to your predictions and the actual scores
def get_accuracy(prediction, target):
    p = prediction
    t = target
    difference = p - t
    correct = np.where(abs(difference) <= 0.5, 1, 0)
    print(correct)
    num_correct = correct.sum()
    total = len(correct)
    print('Accuracy: ')
    print(num_correct / total)


# @cli.command()
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

@cli.command()
def get_var():
    recipes = pd.read_pickle('final_dataframe.pkl')
    print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
    #######################
    recipes.target = recipes['rating']
    print('variance')
    print(recipes.target.std())

@cli.command()
def plot_svr():
    data = pd.read_csv('pred.csv') 
    plt.plot(data.Result, data.Pred, 'bo')
    plt.show()

@cli.command()
def svr():
    recipes = pd.read_pickle('final_dataframe.pkl')
    print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
    #######################


    recipes.target = recipes['rating']
    recipes.data = recipes.drop(['rating'], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.25, random_state=42)
    '''
    svra = SVR(gamma=.001, C=1000, kernel = 'rbf', epsilon = 0.001, verbose = 100)
    print('fitting SVR')
    svra.fit(x_train, y_train)
    y_pred = svra.predict(x_test)
    print('svm score:')
    print(svra.score(x_test, y_test))
    get_accuracy(y_pred, y_test)
    print ("MEAN ABSOLUTE ERROR : " + str(mean_absolute_error(y_test, y_pred)))
    print ("MEAN SQUARED ERROR : " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
    print(svra.get_params())
    d = {'Pred':y_pred, 'Result':y_test}
    df = pd.DataFrame(d)
    df.to_csv('svr_pred.csv')
    '''
    svm.svca = SVC()
    svm.svca.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(metrics.confusion_matrix(y_test, y_pred))  
    print(metrics.classification_report(y_test, y_pred))


    '''
    # GRID SEARCH
    parameters = {'kernel':('linear', 'rbf'), 'C':[.001, .1, 1, 10], 'gamma':[.001, .1, 1, 10, 'scale'], 'epsilon':[0.001, 0.01, 0.1, 1, 10]}
    print('grid search')
    clf = GridSearchCV(svra, parameters, cv=5, verbose = 5, n_jobs = -1)
    print(clf.best_params_) 
    '''

    '''
    x_train1, x_test1, y_train1, y_test1 = train_test_split(recipes.feature, recipes.target, test_size=0.2, random_state=42)
    clf = svm.LinearSVC(class_weight = 'balanced', verbose = 1, max_iter = 200000)
    print('fitting')
    clf.fit(list(x_train1), y_train1)
    print('predict')
    y_pred = clf.predict(list(x_test1))
    print(metrics.confusion_matrix(y_test1,y_pred))  
    print(metrics.classification_report(y_test1,y_pred))  
    
    clf = LogisticRegression(class_weight = 'balanced')
    print('LOGIT fitting')
    clf.fit(list(x_train1), y_train1)
    print('LOGIT predict')
    y_pred = clf.predict(list(x_test1))
    print(metrics.confusion_matrix(y_test1,y_pred))  
    print(metrics.classification_report(y_test1,y_pred))  
    
    #Grid Search
    Cs = [0.1, 1, 10, 100]
    gammas = [0.1, 1, 10, 100]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear', class_weight = 'balanced'), param_grid, verbose = 3)
    grid_search.fit(list(x_train1), y_train1)
    grid_search.best_params_
    print(grid_search.best_params_)
    '''

@cli.command()
def svr_grid_search():
    recipes = pd.read_pickle('final_dataframe.pkl')
    print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
    #######################
    recipes.target = recipes['rating']
    recipes.data = recipes.drop(['rating'], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.25, random_state=42)
    svr_a = SVR()
    params = {'kernel': ['rbf'], 'gamma': [1e-4, 1, 'scale'], 'C': [1000, 10000], 'epsilon':[0.001, 1, 10]}
    print('Grid Search')
    grid_search = GridSearchCV(svr_a, params, scoring = 'neg_mean_absolute_error', n_jobs=-1, iid=True, cv=3, verbose=10)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_) 


@cli.command()
def forest():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    recipes = recipes.dropna(subset=['rating', 'ingredients'])
    recipes['ingredients'] = recipes['ingredients'].apply(removeNums)
    

    recipes.target = recipes['rating']
    recipes.data = recipes.drop(['rating'], axis = 1)

    # #print(recipe_tags.target.head())
    # print(recipe_tags.target.value_counts())
    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)

    #NOW CHANGE FROM VEC OF STRINGS TO ONE FAT STRING
    ingredient_string_test = x_train['ingredients'].apply(lambda x : " ".join(str(word) for word in x))



    print('='*80)
    # get most common bigrams
    # Now this also takes out stop words
    sw = stopwords.words('english')
    sw.append('andor')

    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = sw,   \
                             max_features = 5000) 

    train_data_features = vectorizer.fit_transform(ingredient_string_test)
    vocab = vectorizer.vocabulary_
    train_data_features = train_data_features.toarray()
    count_values = train_data_features.sum(axis=0)
    print('='*80)
    # Make list of 5000 most common words and bigrams
    i = 0
    most_common_words = []
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:5000]:
        print (bg_count, bg_text)
        most_common_words.append(bg_text)
    print('\nVector of most common words and bigrams is this long: ')
    print(len(most_common_words))

    print("DATA FEATURES SHAPE: ", train_data_features.shape)

    print("Training the random forest...")

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestRegressor(n_estimators = 100) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, y_train['rating'])

    #clean valid dataset 
    ingredient_string_test = x_valid['ingredients'].apply(lambda x : " ".join(str(word) for word in x))
    valid_data_features = v.transform(ingredient_string_test)
    valid_data_features = valid_data_features.toarray()
    # Use the random forest to make sentiment label predictions
    result = forest.predict(valid_data_features)
    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

    
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
    x_train, x_test, y_train, y_test = train_test_split(recipe_tags.data, recipe_tags.target, test_size=0.2, random_state=42)

    # Run logistic regression, print results
    lin = LinearRegression()
    lin.fit(x_train, y_train)
    print('Linear Regression:')
    predictions = lin.predict(x_test)
    print('Score:')
    print(lin.score(x_test, y_test),'\n')
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(y_test, predictions))
    print('Coefficients: ' + str(lin.coef_))
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
    train, test = train_test_split(ingred_to_rating, test_size=0.2, random_state=42)
    train_tagged = train.apply(lambda r: TaggedDocument(words=tokenize_text(r['ingredients']), tags=[r.rating]), axis=1)
    test_tagged = test.apply(lambda r: TaggedDocument(words=tokenize_text(r['ingredients']), tags=[r.rating]), axis=1)




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



def removeNums(x):
    # for val in x:
    #     nums = re.search('([0-9]+[,.]?[0-9]*([\/][0-9]+[,.]?[0-9]*)*)', val)
    #     if nums:
    #         print('found a number, ' + str(nums.group(1)) + ' corresponding to : ' + str(val))
    result = [re.sub('[^a-zA-Z ]+', '', val) for val in x]
    return result



@cli.command()
@click.option('--vals', default=100, help='how many of top words to include') #how manhy of the top 5000 words to take ink
@click.option('--tagnum', default = 675, help='how many of top tags to include')
@click.option('--notags', default=False, is_flag=True, help='whether to zip up with the tags')
@click.option('--quant', default=False, is_flag=True, help='whether to augment with quantities')
def finalDF(vals, tagnum, notags, quant):
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

    if quant:
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
    else:
        for i, row in tqdm(ingredients.iterrows()):
            feature_vec = []
            ingred_list = list(row['full_ingredients'])  
            ingred_string = ' '.join(ingred_list)
            for word in most_common_words:
                if word in ingred_string:
                    # print("OLD entered here with word : " + word)
                    feature_vec.append(1)
                else:
                    feature_vec.append(0)
    
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
    print(combined_df['rating'].value_counts())


  
    print("this is the new matrix shape: " + str(combined_df.shape))
    final = combined_df.dropna()
    print(final.head())
    print("this is the final matrix shape without null: " + str(final.shape))
    final.to_pickle('final_dataframe.pkl')

@cli.command()
def rf_grid_search():
    recipes = pd.read_pickle('final_dataframe.pkl')
    print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
    #######################
    recipes.target = recipes['rating']
    recipes.data = recipes.drop(['rating'], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.2, random_state=42)


    random.seed(42)
    rf = RandomForestRegressor()
    params = {'n_estimators':[10, 50, 100], 'max_features':['auto', 'sqrt'], 'max_depth':[10, 100, 1000, None]}
    print('Grid Search')
    grid_search = GridSearchCV(rf, params, scoring = 'neg_mean_squared_error', n_jobs=-1, iid=True, cv=3, verbose=100)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_) 


def relu_advanced(x):
    return K.relu(x, max_value=5, alpha=0)

def round(x):
    x = float(x)
    return math.trunc(x)
    # up = math.ceil(x * 8)/8
    # down = math.floor(x * 8)/8
    # better = min(abs(up - x), abs(down - x))

    # if better == abs(up - x):
    #     return up
    # else:
    #     return down

@cli.command()
@click.option('--epoch', default=50)
@click.option('--drop', is_flag=True, default=False)
def neuralnetfiltered(epoch,drop):
    recipes = pd.read_pickle('final_dataframe.pkl')
    print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
    #######################
    if(drop):
        recipes.drop(['calories', 'protein', 'fat', 'sodium'], axis = 1)
    recipes.target = recipes['rating']
    recipes.data = recipes.drop(['rating'], axis = 1)
    # recipe_filtered.data = recipe_filtered['protein']

    
    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
    #TODO: center the shits (calories)

    # Define model
    model = Sequential()
    model.add(Dense(100, input_dim=len(recipes.columns) - 1, activation= "relu"))
    # model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(50, activation= "relu"))
    # model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation=relu_advanced))
    model.summary() #Print model Summary

    # Compile model
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    
    # Fit Model

    history = model.fit(x_train, y_train, epochs=epoch, batch_size = 20, verbose=1, validation_data = (x_valid, y_valid))
    #TODO: look into tthis stuff so we can use it, visualize in here 
    # print("Training accuracy: " , np.mean(model_output.history['acc']))
    # print("Validation accuracy: " , np.mean(model_output.history['val_acc']))

    # rounding = np.vectorize(round)
    print('='*50)
    prediction_train_nn= model.predict(x_train)

    score = np.sqrt(mean_squared_error(prediction_train_nn, y_train))
    print ("nn TRAIN  root mean square error", score)
    score2 = r2_score(y_train,prediction_train_nn)
    print("nn TRAIN  r2", score2)
    
    correct = 0
    total = 0
    for index, value in y_train.iteritems():
        pred = prediction_train_nn[total][0]
        val = value
        diff = abs(pred - val)
        if(diff <= .75):
            correct += 1
        total += 1
    print("TRAIN accuracy: " + str(float(correct)/float(total)))

    print('='*50)
    prediction_nn= model.predict(x_valid)
    score = np.sqrt(mean_squared_error(prediction_nn,y_valid))
    print ("nn validation root mean square error", score)
    score2 = r2_score(y_valid,prediction_nn)
    print("nn validation r2", score2)
    correct = 0
    total = 0
    for index, value in y_valid.iteritems():
        pred = prediction_nn[total][0]
        val = value
        diff = abs(pred - val)
        if(diff <= .75):
            correct += 1
        total += 1
    print("NEURAL NET VALID accuracy: " + str(float(correct)/float(total)))

    prediction_nn_test= model.predict(x_test)
    score = np.sqrt(mean_squared_error(prediction_nn_test,y_test))
    print ("nn TEST root mean square error", score)
    score2 = r2_score(y_test,prediction_nn_test)
    print("nn test r2", score2)
    correct = 0
    total = 0
    for index, value in y_test.iteritems():
        pred = prediction_nn_test[total][0]
        val = value
        diff = abs(pred - val)
        if(diff <= .75):
            correct += 1
        total += 1
    print("NEURAL NET TEST accuracy: " + str(float(correct)/float(total)))


    print('='*50)
    mean_rating = float(recipes['rating'].mean())
    print(mean_rating)
    print(type(mean_rating))
    print(len(y_valid))
    mean_baseline = np.full(len(y_valid), mean_rating)
    score = np.sqrt(mean_squared_error(mean_baseline,y_valid))
    print ("MEAN BASELINE root mean square error", score)
    score2 = r2_score(y_valid,mean_baseline)
    print("MEAN BASELINE r2", score2)

    correct = 0
    total = 0
    for index, value in y_valid.iteritems():
        pred = mean_rating
        val = value
        diff = abs(pred - val)
        if(diff <= .75):
            correct += 1
        total += 1
    print("MEAN total: " + str(total) + " length : "+ str(len(y_valid)))
    print("MEAN accuracy: " + str(float(correct)/float(total)))




    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    print('='*50)

    print('planting the trees:')
    #random forest code
    random.seed(42)
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,verbose = 5, max_features='sqrt')
    rf.fit(x_train, y_train)
    print("fit to random forest")
    y_valid_rf = rf.predict(x_valid)
    score = np.sqrt(mean_squared_error(y_valid, y_valid_rf))
    print ("random forest mean square", score)
    score2 = r2_score(y_valid, y_valid_rf)
    print("random forest r2", score2)


    print('ONTEST, RANDOM FOREST')
    y_test_rf = rf.predict(x_test)
    score = np.sqrt(mean_squared_error(y_test, y_test_rf))
    print ("random forest mean square", score)
    score2 = r2_score(y_test_rf, y_test)
    print("random forest r2", score2)

    correct = 0
    total = 0
    for index, value in y_valid.iteritems():
        pred = y_test_rf[total]
        val = value
        diff = abs(pred - val)
        if(diff <= .75):
            correct += 1
        total += 1
    print("RANDOM FOREST TEST accuracy: " + str(float(correct)/float(total)))
    correct = 0
    total = 0
    for index, value in y_valid.iteritems():
        pred = y_valid_rf[total]
        val = value
        diff = abs(pred - val)
        if(diff <= .75):
            correct += 1
        total += 1
    print("RANDOM FOREST VALID accuracy: " + str(float(correct)/float(total)))



    print("AS A REMINDER, THIS RAN ON THE FOLLOWING DIMENSIONS: " + str(recipes.shape))


    quit()
     


    # #import Epicurious (tags) dataset
    # se = pd.Series(feature_vectors)
    # recipes['feature'] = se.values
    # # mean_rating = recipes['rating'].mean()
    # # recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)
    
    # center = recipes[['calories', 'protein', 'fat', 'sodium']]
    # print('CENTERED'*50)
    # print(center)
    # print('MEANS'*20)
    # colmeans = center.mean(axis = 0) 
    # print(colmeans)
    # print('MEDIANS' * 20)
    # colmeadians = center.median(axis = 0)
    # print(colmeadians)


    # #take out extraneous vals
    # center['rating'] = recipes['rating']
    # center['rating'] = recipes['rating'].values
    # center_filtered = recipes[ (abs(recipes['calories'] - colmeadians['calories']) < colmeadians['calories']) & (abs(recipes['sodium'] - colmeadians['sodium']) < colmeadians['sodium'])]
    # center_filtered = center_filtered.drop(['title'], axis = 1)
    # print(center_filtered)

    # counter = 0
    # for index, row in center_filtered.iterrows():
    #     if row['calories'] > 6350.682993:
    #         print(index, row['calories'])
    #         counter += 1
    # print('counter ' * 100)
    # print(counter)

    # center_filtered_vals = center_filtered[['calories', 'protein', 'fat', 'sodium', 'feature']]
    # print('MEANS'*20)
    # colmeans_filtered = center_filtered_vals.mean(axis = 0) 
    # print(colmeans_filtered)
    # print('MEDIANS' * 20)
    # colmeadians_filtered = center_filtered_vals.median(axis = 0)
    # print(colmeadians_filtered)

    # mean_centered = center_filtered_vals - colmeans_filtered
    # mean_centered['rating'] = center_filtered['rating']
    # mean_centered['rating'] = center_filtered['rating'].values





# neuralnet implementation riperonies.
@cli.command()
def neuralnet():
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


    print("PRINTING EXTRANEOUS VALS: ")
    for val in y_train:
        if val > 2:
            print(val)
    print("DONE PRINTING EXTRANEOUS VALS")

    random.seed(42)
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(x_train, y_train)
    print("fit to random forest")

    print("This is y_train: ", y_train)
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
    print(pred)
    # quit()
    for index, row in pred.iterrows():
        if abs(row[0] - y_valid.iloc[index][0]) > 5:
            print("pred: ", pred[i], ' and y: ', y_valid[i])
            print("diff: ", abs(pred[i] - y_valid[i]))

    score = np.sqrt(mean_squared_error(y_valid,pred))
    print ("neural network", score)



    #Prediction using Random Forest 
    y_valid_rf = rf.predict(x_valid)
    score = np.sqrt(mean_squared_error(y_valid_rf,y_valid))
    print ("random forest", score)



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
