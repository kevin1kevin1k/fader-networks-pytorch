import numpy as np
import matplotlib.image as mpimg
import skimage
import skimage.transform
from torch.utils.data import Dataset

np.random.seed(1)

class CelebA(Dataset):
    def __init__(self, file_list, file_path, flip_prob = .5):
        self.data = file_list
        self.file_path = file_path
        self.flip_prob = flip_prob
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return_img = mpimg.imread(self.file_path+self.data[idx][0])[:,:,:3]
        # Data augmentation
        if np.random.rand() < self.flip_prob:
            return_img = np.fliplr(return_img)
        # transposed to channel first
        return_img = return_img.transpose(-1,-3,-2)
        # Scaling to -1~+1
        return_img = 2*return_img -1

        attr = np.array(self.data[idx][1])
        return return_img,attr