from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import torchio as tio
import numpy as np
import torch

from thesis_brainage.scripts.prep_data import full_data_load


class loader3D(Dataset):
    """
    Args:
        data_directory: path to the data directory
        clean: boolean to clean the data from single scan and CI participants
        preprocess_cat: boolean to preprocess categorical data
        image_size: size of the input image
        target_name: name of the target variable
        data_directory: path to the data directory
        optional_meta: list of optional metadata features
    """
    
    #args: path to data, image size, target name, clean (remove single scan participants and so on), optional meta data
    def __init__(self, args): 
        self.demo = full_data_load(fp_oasis=args.data_directory, clean=args.clean, preprocess_cat=args.preprocess_cat)
        print(f"Loaded {len(self.demo)} participants from {args.data_directory}")
        
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
            img_dir1 = os.path.join(self.datadir, 'derivatives', 'mriprep', participant_id, session1)
            img_dir2 = os.path.join(self.datadir, 'derivatives', 'mriprep', participant_id, session2)
            pattern1 = os.path.join(img_dir1, '*T1w.nii.gz')
            pattern2 = os.path.join(img_dir2, '*T1w.nii.gz')

            matching_files1 = glob.glob(pattern1)
            matching_files2 = glob.glob(pattern2)

            if not matching_files1 or not matching_files2:
                print(f"Warning: No matching T1w image found for {participant_id} in session(s). Skipping.")
                continue  # or raise an error, depending on your needs
            path1 = matching_files1[0]
            path2 = matching_files2[0]
            self.image_pair_paths.append((path1, path2))

        # Save targets for each pair
        self.targets = self.demo[self.targetname].values

        if len(args.optional_meta)>0:
            self.optional_meta = np.array(self.demo[args.optional_meta]).astype('float32')

        else:
            self.optional_meta = np.array([])


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
            meta = torch.tensor(self.optional_meta[index], dtype=torch.float32)
            return [image1, image2, meta, target]

        else:
            return [image1, image2, target]

        
    def __len__(self):
        return len(self.image_pair_paths)