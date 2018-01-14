import os
import matplotlib.image as mpimg
import skimage
import skimage.transform
import numpy as np


data_path = '/home/alexliu/Downloads/CelebA/Img/img_align_celeba_png.7z/img_align_celeba_png/'
output_path = '/home/alexliu/ADLxMLDS2017/final/data/img/'
tmp = os.listdir(data_path)

def read_img(data_path,f_path):
    img = mpimg.imread(data_path+f_path)[-178:]
    img = skimage.transform.resize(img,(256,256),mode='constant')
    return img

for img in tmp:
    plt.imsave(output_path+img, read_img(data_path,img))