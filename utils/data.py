
"""
Utility classes for datasets which we test/train on
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "17-08-2021"

from torch.utils.data import DataLoader

class DataModule():
    """"""

    def __init__(self, root_dir: str = "dataset", test_ratio: float = 0.2):
        """
        
        Parameters
        ----------
        root_dir: str
            The root dir where the data is located or should be downloaded.
        test_ratio: float
            The percentage of the data that should be assigned to the test set and to the validation set each.
            This is only used for "fixed" split_mode.
        """
        
        


    def make_train_loader(self, batch_size = 64):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def make_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=64, shuffle=False)

    def make_val_loader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)








