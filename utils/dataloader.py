import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor


class DCganDataset(Dataset):
    def __init__(self, train_lines, input_shape):
        super(DCganDataset, self).__init__()

        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.input_shape    = input_shape

    def __len__(self):
        return self.train_batches

    def preprocess_input(self, image, mean, std):
        image = (image/255 - mean)/std
        return image

    def __getitem__(self, index):
        index   = index % self.train_batches
        image   = Image.open(self.train_lines[index].split()[0])
        image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)
        
        image   = np.array(image, dtype=np.float32)
        image   = np.transpose(self.preprocess_input(image, 0.5, 0.5), (2, 0, 1))
        return image

def DCgan_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = np.array(images)
    return images
