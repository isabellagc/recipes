# Works on python 2.7 on my computer.
# Template for optimization. 
# Parameters can be changed further, this is starting point for me.

### HYPEROPT ###

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys

import pandas as pd
import numpy as np

np.random.seed(6669)

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2
import tensorflow as tf
from keras import backend as K
# tf.python.control_flow_ops = tf

# Based on Faron's stacker. Thanks!

# ID = 'id'
# TARGET = 'loss'
# NFOLDS = 5
# SEED = 669
# NROWS = None
# DATA_DIR = "../../"

# TRAIN_FILE = "{0}/train.csv".format(DATA_DIR)
# TEST_FILE = "{0}/test.csv".format(DATA_DIR)
# SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)

# train = pd.read_csv(TRAIN_FILE, nrows=NROWS)
# test = pd.read_csv(TEST_FILE, nrows=NROWS)

# train_indices = train[ID]
# test_indices = test[ID]

# y_train_full = train["loss"]
# y_train_ravel = train[TARGET].ravel()

# train.drop([ID, TARGET], axis=1, inplace=True)
# test.drop([ID], axis=1, inplace=True)

# print("{},{}".format(train.shape, test.shape))


# ntrain = train.shape[0]
# ntest = test.shape[0]
# train_test = pd.concat((train, test)).reset_index(drop=True)


# features = train.columns

# cats = [feat for feat in features if 'cat' in feat]
# for feat in cats:
# train_test[feat] = pd.factorize(train_test[feat], sort=True)[0]

# train = train_test.iloc[:ntrain, :]

# # Using train_test_split in order to create random split for Keras,
# # otherwise it'll use last part of data when
# # validation_split is provided in the model parameters.

# X_train, X_val, y_train, y_val = train_test_split(train, y_train_full, test_size = 0.15)
recipes = pd.read_pickle('final_dataframe.pkl')
print("CURRENT DIMENSIONS WE ARE WORKING WITH: " + str(recipes.shape))
#######################
recipes.drop(['calories', 'protein', 'fat', 'sodium'], axis = 1)
recipes.target = recipes['rating']
recipes.data = recipes.drop(['rating'], axis = 1)
# recipe_filtered.data = recipe_filtered['protein']


X_train, X_test, y_train, y_test = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(recipes.data, recipes.target, test_size=0.3, random_state=42)
#TODO: center the shits (calories)
# print X_train.shape
# print X_val.shape
# print y_train.shape
# print y_val.shape

x_train_array = np.array(X_train, dtype = float)
y_train_array = np.array(y_train)
x_val_array = np.array(X_val, dtype = float)
y_val_array = np.array(y_val)




### GRIDSEARCHCV ###
# Additionally, here implemented with 3 folds.


def relu_advanced(x):
    return K.relu(x, max_value=5, alpha=0)

# x_train_array_full = np.array(train, dtype = float)
# y_train_array_full = np.array(y_train_full, dtype = float)


def neural_network(size1 = 100, size2 = 50, dropout1 = .2, dropout2 = .1, loss1 = 'mean_squared_error', opti = 'adam', activation1='relu', activation2='relu', init1 = 'glorot_normal'):
    model = Sequential()
    model.add(Dense(size1, activation=activation1, input_dim=len(recipes.columns) - 1, kernel_initializer=init1))
    # model.add(PReLU())
    model.add(Dropout(dropout1))

    model.add(Dense(size2, activation=activation1, kernel_initializer=init1))
    # model.add(PReLU())
    model.add(Dropout(dropout2))

    model.add(Dense(1, activation=relu_advanced, kernel_initializer=init1))
    
    model.compile(loss = loss1, optimizer = opti, metrics = ["mean_squared_error"])
    return(model)



#make the grid
NN_grid = KerasRegressor(build_fn=neural_network, verbose = 100)

print('Length of data input: ', y_train.shape[0])

batch_size = [10, 20, 80]
epochs = [30, 50, 100]
size1 = [100, 200, 300]
size2 = [10, 50, 100]
# dropout2 = [.1, .2]
# dropout1 = [.4, .3, .2]

param_grid = dict(batch_size=batch_size, epochs=epochs, size1=size1, size2=size2)
validator = GridSearchCV(estimator = NN_grid, param_grid = param_grid, cv = 3, n_jobs = 3, verbose=100)
         




grid_result = validator.fit(X_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

