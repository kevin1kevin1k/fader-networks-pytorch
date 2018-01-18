from src.celeba_dataset import CelebA
import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.celeba_dataset import CelebA

import matplotlib.pyplot as plt

use_gpu = True
num_test_fig = 10
IMAGE_SIZE = 256
num_tags = 2

min_attr = 2.0
max_attr = 2.0
n_interpolations = 4

data_path = 'data/img/'
spliting_path = 'data/list_eval_partition.txt'
attribute_path = 'data/list_attr_celeba.txt'
target_attr = 'Male'
save_fig_path = 'fig/fig_2.jpg'

# Read the Attribute table & split dataset
attri_table = pd.read_csv(attribute_path,sep=' * ',skiprows=1)
split = pd.read_csv(spliting_path,header=None,sep=' ',index_col=0)
split = split.rename(columns={1:'Set'}).join(attri_table[target_attr])
split[target_attr] = split[target_attr]==1



# Read the spliting table & split dataset
test_set = [(idx.split('.')[0]+'.png',[1*row[target_attr],1*(not row[target_attr])]) 
             for idx,row in split.loc[split['Set']==2].iterrows()]

# Create Dataset
debug_set = DataLoader(CelebA(test_set[:num_test_fig],data_path), 
                       batch_size=num_test_fig, shuffle=False, num_workers=1, drop_last=False)
del attri_table
del split

# Load Model
enc = torch.load('checkpoint/FaderNet_Male_250000.enc')
dec = torch.load('checkpoint/FaderNet_Male_250000.dec')

test_img = Variable(torch.zeros(num_test_fig, 3, IMAGE_SIZE, IMAGE_SIZE).float(),requires_grad=False)
test_attr = Variable(torch.zeros(num_test_fig, num_tags),requires_grad=False)

if use_gpu:
    enc = enc.cuda()
    dec = dec.cuda()
    test_img = test_img.cuda()
    test_attr = test_attr.cuda()

# Interpolate Attributes
alphas = np.linspace(1 - min_attr, max_attr, n_interpolations)
alphas = [[1 - alpha, alpha] for alpha in alphas]


# Run test
flp_set = []
for (batch_img,batch_attr) in debug_set:
    test_img.data.copy_(batch_img)
    test_attr.data.copy_(torch.cuda.FloatTensor(alphas[0]))
    reconstruct_img = dec(enc(test_img),test_attr)
    for alpha in alphas:
        test_attr.data.copy_(torch.cuda.FloatTensor(alpha))
        flipped_img = dec(enc(test_img),test_attr)
        tmp = [t for t in (1+flipped_img.cpu().data.numpy().transpose(-4,-2,-1,-3))/2]
        flp_set.append(np.concatenate(tmp,axis=-2))

# Show result
tmp = []
src_image = [(1+img)/2 for img in test_img.cpu().data.numpy().transpose(-4,-2,-1,-3)]
rec_image = [(1+img)/2 for img in reconstruct_img.cpu().data.numpy().transpose(-4,-2,-1,-3)]
tmp.append(np.concatenate(src_image,axis=-2))
tmp.append(np.concatenate(rec_image,axis=-2))
tmp.append(np.concatenate(flp_set,axis=-3))
tmp = np.squeeze(np.concatenate(tmp,axis=-3))
plt.imsave(save_fig_path,tmp)
print('Result saved to',save_fig_path)
#plt.savefig('fig/fig_2.jpg',bbox_inches='tight')
plt.close()
