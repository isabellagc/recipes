from torch.utils.data import DataLoader
from .ExampleDataset import ExampleDataset

##UNDER CONSTRUCTION TODO: put in the right folder location, fix getFullDF so it is relevant
# See below for documentation on dataloaders
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html

def get_full_df(columns_to_keep=['Target Business Description', 'Acquirer Business Description']):
    """Merge dataframes, then drops buybacks, exchange offers, and recaps, and
    encodes merger/acquisition as 1 and withdrawn as 0.
    """
    columns_to_keep = columns_to_keep + ["Encoding", "Form of the Transaction"]

    def encode_completed_or_withdrawn(item):
        deal_completed = 'ithdrawn' not in item
        if deal_completed:
            return 1
        return 0

    print("Merging Excel files...")
    full_df = merge_dfs()

    print("Encoding transactions...")

    # Drop anything that doesn't have "Merger" or "Acquisition" in it
    full_df = full_df[
        full_df['Form of the Transaction'].str.contains("Acquisition") |
        full_df['Form of the Transaction'].str.contains("Merger")
    ]

    # Encode any deals that have the string "ithdrawn" as 0
    full_df['Encoding'] = full_df['Deal Status'].apply(
        encode_completed_or_withdrawn
    )

    # Return df with columns to keep
    return full_df[columns_to_keep]


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
