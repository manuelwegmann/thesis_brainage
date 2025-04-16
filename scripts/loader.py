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

    def __init__(self, args, trainvaltest): 
        self.demo = full_data_load() #fix full_data_load to be generalized
        #no self-augmentation, correct?

        #See  image-directory part in LILAC for to do, implement below

        


