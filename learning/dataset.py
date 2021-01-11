import torch as th
import pandas as pd

class Dataset(th.utils.data.Dataset):
    def __init__(self, filename, targetCol = None):
        dataset = pd.read_csv(filename)

        # If no label Column is specified, take the last column by default
        if targetCol is None:
            targetCol = dataset.columns.values.tolist()[-1]

        # dataset = dataset.sample(frac=1).reset_index(drop=True) # Shuffle
        targets = dataset[targetCol]
        dataset.drop(targetCol, axis = 1, inplace = True) # Drop label to get features
        
        self.targets = targets
        self.features = dataset
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get the item at specified index
        """
        if th.is_tensor(idx):
            idx = idx.tolist()
        
        return th.from_numpy(self.features.iloc[idx].to_numpy()).type(th.FloatTensor),\
             th.tensor(self.targets[idx]).type(th.LongTensor)
