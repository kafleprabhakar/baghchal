import torch as th
import pandas as pd

class Dataset(th.utils.data.Dataset):
    def __init__(self, filename, featuresCols, targetCol):
        dataset = pd.read_csv(filename)
        
        self.targets = dataset.iloc[:, targetCol]
        self.features = dataset.iloc[:, featuresCols]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get the item at specified index
        """
        if th.is_tensor(idx):
            idx = idx.tolist()
        
        return th.from_numpy(self.features.iloc[idx].to_numpy()).type(th.FloatTensor),\
             th.tensor(self.targets.iloc[idx]).type(th.LongTensor)

