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

import json

# Custom defined
# from models import *
from utils import data_utils, test_utils


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


@cli.command()
def vectorize():
    json_recipes = get_json_recipes()
    recipes = pd.DataFrame.from_dict(json_recipes, orient='columns')
    
    # Not tags
    recipe_tags = recipes.drop(['title', 'calories', 'protein', 'fat', 'sodium'], axis = 1)
    mean_rating = recipes['rating'].mean()
    #creating new column where ratings are 1 for good and 0 for bad
    recipes['target'] = np.where(recipes['rating']>=mean_rating, 1, 0)

    print("MEAN RATING: "  + str(mean_rating))
    print str(recipes.head())
    print "COLUMNS" + str(recipes.columns)

    #FOR JOSEPH
    ingred_to_rating = pd.concat((recipes['ingredients'], recipes['rating'], recipes['target']), axis=1, keys=['ingredients', 'rating', 'target'])
    print ('=' * 100)
    print "dataframe for joseph"
    print ('=' * 100)
    print str(ingred_to_rating.head())
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

    



    # Counter = Counter(split_it)
    # most_occur = Counter.most_common(2000) # TODO: change number
    # most_common_words = [word for word, word_count in most_occur]
    # with open('dict.csv', 'w') as f:
    #     for item in most_common_words:
    #         f.write("%s\n" % (str(item)))

    # #create bag-of-words style vectors for training and test sets, 
    # with open('train_vec.csv', 'w') as f:
    #     clean_ing = pd.read_csv('something.csv') # read in list of 
    #     for ingredient in clean_ing['content'].tolist():
    #         l = [];
    #         for word in most_common_words:
    #             if str(word) in str(cont):
    #                 l.append(1)
    #             else:
    #                 l.append(0)
    #         f.write("0")
    #         for word in l:
    #             f.write(",%s" % (word))
    #         f.write("\n")





#import Epicurious (tags) dataset
recipes =pd.read_csv("data/epicurious/epi_r.csv").dropna()




# #IGNORE ALL BELOW UNTIL WE DECIDE WHETHER WE ARE USING THIS FRAMEWORK
# def _train(model, device, lr,
#            train_dataloader, val_dataloader, test_dataloader,
#            epochs, model_save_path):
#     criterion = torch.nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     loss_curve_info = []
#     for epoch in range(epochs):
#         for batch_x, batch_y in train_dataloader:
#             batch_x, batch_y = batch_x.to(device), batch_y.to(device)

#             # Forward Propagation
#             y_pred = model.predict(batch_x)

#             # Compute and print loss
#             loss = criterion(y_pred, batch_y.float())

#             # Zero the gradients
#             optimizer.zero_grad()

#             # perform a backward pass (backpropagation)
#             loss.backward()

#             # Update the parameters
#             optimizer.step()

#         acc = test_utils.batch_accuracy(model, val_dataloader)
#         is_better = len(loss_curve_info) == 0 or \
#             acc > max(
#                 loss_curve_info,
#                 key=lambda x: x['val_acc'])['val_acc']

#         if (is_better):
#             print("Saving model...")
#             model.save(model_save_path + "/model.bin")

#             torch.save(
#                 optimizer.state_dict(),
#                 model_save_path + "/model.bin.optim"
#             )

#         loss_curve_info.append(
#             {"epoch": epoch, "loss": loss.item(), "val_acc": acc}
#         )
#         print('epoch: ', epoch, ' loss: ', loss.item(), 'acc: ', acc)

#     acc = test_utils.batch_accuracy(model, test_dataloader)
#     print('Test accuracy: ', acc)

#     loss_curve_info = pd.DataFrame(loss_curve_info)
#     loss_curve_info.to_csv(
#         "charts/" + str(int(time.time())) + ".csv",
#         index=False
#     )


# @cli.command()
# @click.option('--batch_size', default=10)
# @click.option('--hidden', default=50)
# @click.option('--lr', default=0.001)
# @click.option('--save_to', default="saved_models/", help="dir for models")
# @click.option('--epochs', default=10, help="number of epochs")
# def train(batch_size, hidden, lr, save_to, use_small, epochs):
#     """Run the training loop.

#     Usage: python main.py train [flags]
#     """
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")

#     ###############################################
#     # LOAD DATA
#     ###############################################
#     print("=" * 50)
#     datasets, dataloaders = \
#         data_utils.get_train_test_dataloaders(batch_size)
#     train_dataset, val_dataset, test_dataset = datasets
#     train_dataloader, val_dataloader, test_dataloader = dataloaders

#     ###############################################
#     # LOAD MODEL
#     ###############################################
#     print("=" * 50)
#     model = None  # import or construct model
#     model = model.to(device)

#     ###############################################
#     # START TRAINING
#     ###############################################
#     print("=" * 50)
#     _train(model, device, lr,
#            train_dataloader, val_dataloader, test_dataloader,
#            epochs, save_to)


# @cli.command()
# @click.option('--saved_to', default="saved_models/", help="dir for models")
# def test(hidden, use_model, lr, saved_to):
#     """Run the model on test data if applicable.

#     Usage: python main.py test [flags]
#     """
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     raise NotImplementedError("TODO")

if __name__ == "__main__":
    cli()
