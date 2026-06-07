from torch.utils.data import Dataset

class WordleDataset(Dataset):
    
    def __init__(self, file="words.txt"):
        self.words = []

        with open(file) as f:
            self.words = [line.strip() for line in f]
        
    
    def __getitem__(self, index):
        return self.words[index];

    def __len__(self):
        return self.words.__len__()
            