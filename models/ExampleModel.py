import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleModel(nn.Module):
    def __init__(self, arg):
        super(ExampleModel, self).__init__()
        self.arg = arg

        self.projection = nn.Linear(10, 5, bias=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        pass

    def predict(self, x, eval_mode=False):
        if (eval_mode):
            self.eval()

        out = self.forward(x)

        if (eval_mode):
            raise NotImplementedError("Decide how you want to choose labels")
            predict_labels = torch.max(out, 1)[1]
            self.train(mode=True)
            return predict_labels

        return out

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        raise NotImplementedError("not implemented")
        params = {}  # fill this out
        torch.save(params, path)

