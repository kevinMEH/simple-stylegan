from os import listdir
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class FFHQ(Dataset):
    def __init__(self, transform, path = "./data/ffhq", file_prefix = "", file_postfix = ".png"):
        self.transform = transform
        self.path = Path(path)
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix
        self.length = len(listdir(path))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if(index >= 9000): index += 2000
        padded_index = str(index).rjust(5, "0")
        file_name = self.file_prefix + padded_index + self.file_postfix
        file_path = Path.joinpath(self.path, file_name)
        image = Image.open(file_path)
        image = self.transform(image)
        return image