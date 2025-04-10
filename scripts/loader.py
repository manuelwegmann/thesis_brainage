from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import torchio as tio
import numpy as np

class loader3D(Dataset):

    def __init__(self, args, trainvaltest):
        

