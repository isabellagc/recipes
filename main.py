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


import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier     
from sklearn import preprocessing

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

from sklearn.svm import SVR
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
    correct = np.where(abs(difference) <= 0.75, 1, 0)
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
    # Make list of 5000 most common words and bigrams
    i = 0
    most_common_words = []
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:100]:
        print (bg_count, bg_text)
        most_common_words.append(bg_text)
    print('\nVector of most common words and bigrams is this long: ')
    print(len(most_common_words))

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
    print(recipes.shape)
    
    recipes['feature'] = feature_vectors
    '''
    x_train1, x_test1, y_train1, y_test1 = train_test_split(recipes.feature, recipes.rating, test_size=0.2, random_state=42)
    svra = SVR(gamma='scale', C=1.0, epsilon=0.2, verbose = 3)
    parameters = {'kernel':('linear', 'rbf'), 'C':[.001, .1, 1, 10], 'gamma':[.001, .1, 1, 10, 'scale']}
    print('grid search')
    clf = GridSearchCV(svra, parameters, cv=5, verbose = 5, n_jobs = -1)
    print('fitting SVR')
    clf.fit(list(x_train1), y_train1)
    y_pred = svra.predict(list(x_test1))
    print('svm score:')
    get_accuracy(y_pred, y_test1)
    print ("MEAN ABSOLUTE ERROR : " + str(mean_absolute_error(y_test1, y_pred)))
    print ("MEAN SQUARED ERROR : " + str(np.sqrt(mean_squared_error(y_test1, y_pred))))
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


@cli.command()
@click.option('--vals', default=100, help='how many of top words to include') #how manhy of the top 5000 words to take ink
def finalDF(vals):
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
    for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)[0:vals]:
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
    final.to_pickle('final_dataframe.pkl')



@cli.command()
@click.option('--epoch', default=100)
def neuralnetfiltered(epoch):
    recipes = pd.read_pickle('final_dataframe.pkl')
    print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
    #######################


    recipes.target = recipes['rating']
    recipes.data = recipes.drop(['rating'], axis = 1)
    # recipe_filtered.data = recipe_filtered['protein']

    
    x_train, x_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
    #TODO: center the shits (calories)

    # Define model
    model = Sequential()
    model.add(Dense(50, input_dim=778, activation= "relu"))
    model.add(Dense(50, activation= "relu"))
    model.add(Dense(1))
    model.summary() #Print model Summary

    # Compile model
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    
    # Fit Model
    model_output = model.fit(x_train, y_train, epochs=epoch, batch_size = 20, verbose=1, validation_data = (x_valid, y_valid))
    #TODO: look into tthis stuff so we can use it, visualize in here 
    # print("Training accuracy: " , np.mean(model_output.history['acc']))
    # print("Validation accuracy: " , np.mean(model_output.history['val_acc']))

    prediction_nn= model.predict(x_valid)
    score = np.sqrt(mean_squared_error(prediction_nn,y_valid))
    print ("neural network mean square", score)
    score2 = r2_score(y_valid,prediction_nn)
    print("neural network r2", score2)


    #random forest code
    random.seed(42)
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(x_train, y_train)
    print("fit to random forest")
    y_valid_rf = rf.predict(x_valid)
    score = np.sqrt(mean_squared_error(y_valid, y_valid_rf))
    print ("random forest mean square", score)
    score2 = r2_score(y_valid, y_valid_rf)
    print("random forest r2", score2)


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





    #import Epicurious (tags) dataset
    recipes = pd.read_csv('data/epicurious/epi_r.csv').dropna()



    # Not tags
    recipe_filtered= recipes.drop(['calories', 'protein', 'fat', 'sodium', 'title'], axis = 1)
    print(recipe_filtered)
    recipe_tags = recipe_filtered.drop(['rating'], axis = 1)
    print('TAGS ' * 20)
    print(recipe_tags)
    print('----'*20)
    print('----'*20)
    # Create data and target, split into train and test, I think we should sample evenly

    recipe_filtered.target = recipe_filtered['rating']
    recipe_filtered.data = recipe_filtered.drop(['rating'], axis = 1)
    # recipe_filtered.data = recipe_filtered['protein']

    print(recipe_filtered.target)

    x_train, x_test, y_train, y_test = train_test_split(recipe_filtered.data, recipe_filtered.target, test_size=0.3, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(recipe_filtered.data, recipe_filtered.target, test_size=0.3, random_state=42)


    # Define model
    model = Sequential()
    model.add(Dense(100, input_dim=674, activation= "relu"))
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(1))
    model.summary() #Print model Summary

    # Compile model
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    
    # Fit Model
    # model.fit(x_train, y_train, epochs=10)
    model_output = model.fit(x_train, y_train, epochs=100, batch_size = 20, verbose=1, validation_data = (x_valid, y_valid))
    # print("Training accuracy: " , np.mean(model_output.history['acc']))
    # print("Validation accuracy: " , np.mean(model_output.history['val_acc']))

    pred= model.predict(x_valid)
    score = np.sqrt(mean_squared_error(pred,y_valid))
    print ("neural network mean square", score)
    score2 = r2_score(y_valid,pred)
    print("neural network r2", score2)



    #random forest code:
    random.seed(42)
    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(x_train, y_train)
    print("fit to random forest")
    y_valid_rf = rf.predict(x_valid)
    score = np.sqrt(mean_squared_error(y_valid, y_valid_rf))
    print ("random forest mean square", score)
    score2 = r2_score(y_valid, y_valid_rf)
    print("random forest r2", score2)





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
