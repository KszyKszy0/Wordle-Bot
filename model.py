import torch

class WordleModel(torch.nn.Module):

    def __init__(self, hidden_size = 400):
        super().__init__()

        self.linear1 = torch.nn.Linear(26 * 5 * 3, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, 3200)
        self.softmax = torch.nn.Softmax(dim=0)
        pass

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x