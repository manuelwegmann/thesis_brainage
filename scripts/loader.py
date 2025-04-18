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
            participant_id = str(row['participant_id'])
            path_sessions = os.path.join(self.datadir, participant_id, 'sessions.tsv')
            sessions_file = pd.read_csv(path_sessions, sep='\t')
            session1 = str(sessions_file.iloc[0]['session_id'])
            session2 = str(sessions_file.iloc[-1]['session_id'])
            fname1 = f"{participant_id}_{session1}_run-01_T1w.nii.gz"
            fname2 = f"{participant_id}_{session2}_run-01_T1w.nii.gz"
            path1 = os.path.join(self.datadir, participant_id, session1, 'anat', fname1)
            path2 = os.path.join(self.datadir, participant_id, session2, 'anat', fname2)
            self.image_pair_paths.append((path1, path2))

        # Save targets for each pair
        self.targets = self.demo[self.targetname].values

        if len(args.optional_meta)>0:
            self.optional_meta = np.array(self.demo[args.optional_meta])
        else:
            self.optional_meta = ''


    def __getitem__(self, index):
        target = self.demo[self.targetname].iloc[index]


        path1, path2 = self.image_pair_paths[index]
        image1 = tio.ScalarImage(path1)
        image2 = tio.ScalarImage(path2)
        resize = tio.transforms.Resize(tuple(self.image_size))
        image1 = resize(image1)
        image2 = resize(image2)


        image1 = image1.numpy().astype('float')
        image2 = image2.numpy().astype('float')

        if len(self.optional_meta) > 0:
            meta = self.optional_meta[index, :]
            return [image1, image2, meta, target]

        else:
            return [image1, image2, target]

        
    def __len__(self):
        return len(self.image_pair_paths)

