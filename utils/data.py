
"""
Utility classes for datasets which we test/train on
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "17-08-2021"

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision
import torch

import os

import numpy as np
from PIL import Image


class DirectoryDataset(Dataset):

    def __init__(self, dirs, root_dir) -> None:
        super().__init__()

        self.first_transform = torchvision.transforms.Compose([
            transforms.Resize((90, 90)),
            transforms.ToTensor()
        ])
        
        sample_list = []
        for dir in dirs:
            subdir = os.path.join(root_dir, dir)
            samples = os.listdir(subdir)
            for sample in samples:
                sample_list.append(os.path.join(subdir, sample))

        self.samples = np.array(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        pil_img = Image.open(self.samples[index])
        x = torch.squeeze(self.first_transform(pil_img), dim = 0)#add transforms (image augmentation)
        input = torch.ones_like(x) * -1
        
        left_margin = np.random.randint(5, 11)
        left_margin_size = np.random.randint(75 + left_margin, 86)
        top_margin = np.random.randint(5, 11)
        top_margin_size = np.random.randint(75 + top_margin, 86)

        input[top_margin :top_margin_size, left_margin : left_margin_size] = x[top_margin :top_margin_size, left_margin : left_margin_size]

        return x, input, (left_margin, left_margin_size, top_margin, top_margin_size)

class TestDataset(Dataset):
    def __init__(self, testset_file_path) -> None:
        super().__init__()

        self.first_transform = torchvision.transforms.Compose([
            transforms.ToTensor()
        ])
        
        testset = np.load(testset_file_path, allow_pickle=True)

        self.input_arrays = np.array(testset["input_arrays"])
        self.known_arrays = np.array(testset["known_arrays"])
        self.sample_ids = np.array(testset["sample_ids"])

    def __len__(self):
        return len(self.input_arrays)
    
    def __getitem__(self, index):
        input_array, known_array = self.input_arrays[index], self.known_arrays[index]

        input = self.first_transform(input_array)
        known_array = torch.tensor(known_array, dtype=torch.bool)
        input[~known_array] = -1

        return input, self.sample_ids[index]




class DataModule():
    """"""

    def __init__(self, root_dir: str = "dataset", test_ratio: float = 0.2):
        """
        
        Parameters
        ----------
        root_dir: str
            The root dir where the data is located or should be downloaded.
        test_ratio: float
            The percentage of the data-folders that should be assigned to the test set and to the validation set each.
            This is only used for "fixed" split_mode.
        """
        #Split dirs and not samples because samples are not identically distributed across directories
        self.dirs = np.array(os.listdir(root_dir))
        
        n_testset_dirs = int(len(self.dirs) * test_ratio)
        indices = range(len(self.dirs))


        self.test_dataset = DirectoryDataset(self.dirs[indices[n_testset_dirs:n_testset_dirs * 2]], root_dir)
        self.val_dataset = DirectoryDataset(self.dirs[indices[:n_testset_dirs]], root_dir)
        self.train_dataset = DirectoryDataset(self.dirs[indices[n_testset_dirs * 2:]], root_dir)



    def make_train_loader(self, batch_size = 64):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def make_test_loader(self, batch_size = 64):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

    def make_val_loader(self, batch_size = 64):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)








