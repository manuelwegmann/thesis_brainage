from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import torchio as tio
import numpy as np
from .prep_data import full_data_load

class loader3D(Dataset):
    
    #args: path to data, image size
    def __init__(self, args): 
        self.demo = full_data_load() #fix full_data_load to be generalized
        #no self-augmentation, correct?

        #See  image-directory part in LILAC for to do, implement below
        id_unique = np.unique(self.demo['participant_id'])
        index_image_combination = np.empty((0, 2))
        for id in id_unique:
            indices = np.where(self.demo['participant_id'] == id)[0]
            ### from here work in progress
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_image_combination = np.append(index_image_combination, indices[tmp_combination], 0)





        #resize images
        self.image_size = args.image_size 


        


