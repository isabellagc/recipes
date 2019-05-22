import torch
from torch.utils.data import Dataset


# See below for documentation
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset


class ExampleDataset(Dataset):
    def __init__(self, arg):
        super(ExampleDataset, self).__init__()
        self.arg = arg

        # relevant preprocessing here
        self.x
        self.y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        raise NotImplementedError()

    def save(self, name):
        raise NotImplementedError()
