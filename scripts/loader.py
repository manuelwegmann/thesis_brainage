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
    
    #args: path to data, image size, target name, optional meta data
    def __init__(self, args): 
        self.demo = full_data_load() #fix full_data_load to be generalized and do all the necessary steps
        #no self-augmentation, correct?
        
        self.image_size = args.image_size #resize images
        self.targetname = args.target_name #save target for training
        self.datadir = args.data_directory  #save data directory

        # Build file path pairs
        self.image_pair_paths = []
        for _, row in self.demo.iterrows():
            path_sessions = os.path.join(self.datadir, str(row['participant_id']), 'sessions.tsv')
            sessions_file = pd.read_csv(path_sessions, sep='\t')
            session1 = str(sessions_file.iloc[0]['session_id'])
            session2 = str(sessions_file.iloc[-1]['session_id'])
            path1 = os.path.join(self.datadir, row['participant_id'], session1)
            path2 = os.path.join(self.datadir, row['participant_id'], session2)
            self.file_pairs.append((path1, path2))

        # Save targets for each pair
        self.targets = self.demo[self.targetname].values

        if len(args.optional_meta)>0:
            self.optional_meta = np.array(self.demo[args.optional_meta])
        else:
            self.optional_meta = ''


    def __getitem__(self, index):
        target = self.demo[self.targetname][index]

        if len(self.optional_meta) > 0:
            meta = self.optional_meta[index, :]

        file_path_sessions = os.path.join(self.datadir, str(self.demo['participant_id'].iloc[index]), 'sessions.tsv')
        sessions_file = pd.read_csv
        fname2 = os.path.join(self.imgdir, self.demo.fname.iloc[int(index2)])

        image1 = tio.ScalarImage(fname1)
        image2 = tio.ScalarImage(fname2)

        resize = tio.transforms.Resize(tuple(self.image_size))
        image1 = resize(image1)
        image2 = resize(image2)

        if self.augmentation:
            pairwise_transform_list = []
            imagewise_transform_list = []

            if np.random.randint(0, 2):

                if not self.jobname == 'oasis-aging':# oasis-aging dataset w/o affine transform
                    if np.random.randint(0, 2):
                        affine_degree = tuple(np.random.uniform(low=-40, high=40, size=3))
                        affine_translate = tuple(np.random.uniform(low=-10, high=10, size=3))
                        pairwise_transform_list.append(tio.Affine(scales=(1, 1, 1),
                                                                  degrees=affine_degree,
                                                                  translation=affine_translate,
                                                                  image_interpolation='linear',
                                                                  default_pad_value='minimum'))

                if np.random.randint(0, 2):
                    pairwise_transform_list.append(tio.Flip(axes=('LR',)))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomNoise(mean=0, std=2))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomGamma(0.3))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomBlur(2))

            if len(pairwise_transform_list) > 0:
                pairwise_augmentation = tio.Compose(pairwise_transform_list)
                image1 = pairwise_augmentation(image1)
                image2 = pairwise_augmentation(image2)

            if len(imagewise_transform_list) > 0:
                imagewise_augmentation = tio.Compose(imagewise_transform_list)
                image1 = imagewise_augmentation(image1)
                image2 = imagewise_augmentation(image2)

        image1 = image1.numpy().astype('float')
        image2 = image2.numpy().astype('float')

        if len(self.optional_meta) > 0:
            return [image1, target1, meta1], \
                   [image2, target2, meta2]

        else:
            return [image1, target1], \
                   [image2, target2]


        


