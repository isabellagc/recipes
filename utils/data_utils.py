from torch.utils.data import DataLoader
from .ExampleDataset import ExampleDataset

#delete unecessary later
import os
import gensim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# from .vocab import VocabEntry
# from .TransactionDataset import TransactionDataset


##UNDER CONSTRUCTION TODO: put in the right folder location, fix getFullDF so it is relevant

# See below for documentation on dataloaders
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html
def get_full_core_df(columns_to_keep=['recipe_id','user_id','rating'], version='train_rating'):
    """Merge dataframes, then drops buybacks, exchange offers, and recaps, and
    encodes merger/acquisition as 1 and withdrawn as 0.
    """
    columns_to_keep = columns_to_keep 

    folder = 'data/foodrecsysv1/'
    file = 'core-data-'+ version + '.csv'
    df = pd.read_csv(folder + '/' + file)

    return df

    print("Encoding transactions...")


#BELOW HELPFUL if we have a full dataset not split.... the foodsys one is! 
#so use later....
def get_split_dfs(split=(0.8, 0.1, 0.1), random_seed=1337):
    """Calls `get_full_df` and then breaks it up into train/val/test splits"""
    # Enforce train/val/test or train/test splits
    assert len(split) == 3 or len(split) == 2

    # get and shuffle the dataset
    full_df = get_full_df()
    full_df = full_df.sample(frac=1, random_state=random_seed)

    # make the split
    num_items = len(full_df)
    train_cutoff = int(split[0] * num_items)
    val_cutoff = int((split[0] + split[1]) * num_items)
    train_df = full_df.iloc[:train_cutoff]
    val_df = full_df.iloc[train_cutoff:val_cutoff]
    test_df = full_df.iloc[val_cutoff:]

    if len(split) == 2:
        # if split is train/val/test, the last
        return (train_df, val_df)

    return (train_df, val_df, test_df)

def get_train_test_dataloaders(batch_size):
    # create Dataset objects
    train_df, val_df, test_df = get_split_dfs(random_seed=random_seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    datasets = (train_dataset, val_dataset, test_dataset)
    return datasets, dataloaders
