"""
This script contains a reworked data loader that allows for all available image pairs of a participant to be loaded.
"""


from torch.utils.data import Dataset
import pandas as pd
import os
import glob
import torchio as tio
import numpy as np
import torch

from prep_data import add_classification, exclude_CI_participants, exclude_single_scan_participants, check_folders_exist

def load_participants(folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3', clean = True, add_age = False):
    """
    Input:
        folder_path: path to the folder containing the participants.tsv file
        clean: boolean to clean the data from single scan and CI participants
        add_age: whether to add age
    Output:
        df: dataframe with the participants and their gender (possibly age)
    """
    participants_file_path = os.path.join(folder_path, 'participants.tsv')
    df = pd.read_csv(participants_file_path, sep='\t')
    df = check_folders_exist(df, folder_path) #delete participants that do not have a folder
    df = add_classification(df, folder_path) #add classification to the dataframe
    if clean: # Exclude participants with CI
        df = exclude_CI_participants(df)
    if add_age:
        filtered_rows = []
        for _, row in df.iterrows():
            participant_id = str(row['participant_id'])
            sessions_file_path = os.path.join(folder_path, participant_id, 'sessions.tsv')

            if os.path.exists(sessions_file_path):
                sessions_file = pd.read_csv(sessions_file_path, sep='\t')
                age_values = sessions_file['age'].dropna()
                if not age_values.empty:
                    row['age'] = age_values.iloc[0]
                    filtered_rows.append(row)

        df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        return df[['participant_id', 'sex', 'age']]

    else:
        return df[['participant_id', 'sex']]
    


def build_participant_block(participant_id, sex ,folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    """
    Function to extract all available images of one participant. Encode gender as one-hot encoding.
    Input:
        participant_id: id of the participant
        sex: gender of the participant
        folder_path: path to the folder containing all participant folders
    Output:
        Dataframe where each row is a pair of images from the same participant, preprocessed.
    """
    #one-hot encoding
    sex_M = 0
    sex_F = 0
    sex_U = 0
    if sex == 'M':
        sex_M = 1
    if sex == 'F':
        sex_F = 1
    if sex not in ['M', 'F']:
        sex_U = 1
    #load the sessions file for extracting sessions
    sessions_file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')

    if not os.path.exists(sessions_file_path):
        print(f"Warning: The sessions file for participant {participant_id} does not exist.")
        return None
    
    sessions_file = pd.read_csv(sessions_file_path, sep='\t')
    num_sessions = sessions_file.shape[0]
    
    scan_list = []
    age_list = []
    
    #extract sessions and age   
    for i in range(num_sessions-1):
        scan_id = sessions_file.iloc[i]['session_id']
        scan_session = sessions_file[sessions_file['session_id'] == scan_id]
        age = scan_session.iloc[0]['age']
        if np.isnan(age): #skip if age is not available
            break
        scan_list.append(scan_id)
        age_list.append(age)

        return pd.DataFrame({
            'participant_id': [participant_id] * len(scan_list),
            'sex_M': [sex_M] * len(scan_list),
            'sex_F': [sex_F] * len(scan_list),
            'sex_U': [sex_U] * len(scan_list),
            'age': age_list,
            'session_id': scan_list
        })




class loader3D(Dataset):
    """
    Args:
        participant_df: dataframe with basic participant data (ids and gender)
        data_directory: path to the data directory
        image_size: size of the input image
        target_name: name of the target variable
        optional_meta: list of optional metadata features
    """
    
    def __init__(self, args, participant_df):

        #store all blocks of pairs from one participant
        blocks = []

        df = participant_df 

        for _, row in df.iterrows():
            participant_id = str(row['participant_id'])
            sex = str(row['sex'])
            block = build_participant_block(participant_id, sex, folder_path=args.data_directory)
            if block is not None:
                blocks.append(block)

        # concatenate all blocks into one dataframe
        self.demo = pd.concat(blocks, ignore_index=True)
        
        self.image_size = args.image_size #resize images
        self.resize = tio.transforms.Resize(tuple(self.image_size)) #safe resize transform
        self.targetname = args.target_name #save target for training
        self.datadir = args.data_directory  #save data directory

        # Build file path pairs
        self.image_paths = []
        valid_demo_rows = []
        for _, row in self.demo.iterrows():
            participant_id = str(row['participant_id'])
            session = str(row['session_id'])
            img_dir = os.path.join(self.datadir, 'derivatives', 'mriprep', participant_id, session)
            pattern = os.path.join(img_dir, '*T1w.nii.gz')

            matching_files = glob.glob(pattern)

            if not matching_files:
                print(f"Warning: No matching T1w image found for {participant_id} in session(s). Skipping.")
                continue #skip if no matching files are found
            path = matching_files[0]
            self.image_paths.append(path)
            valid_demo_rows.append(row)

        self.demo = pd.DataFrame(valid_demo_rows).reset_index(drop=True)
        self.targets = self.demo[self.targetname].values

        #check length of loaded data
        print(f"Loaded {len(self.image_paths)} images.")
        print(f"Loaded {len(self.demo)} rows of metadata.")
        print(f"Loaded {len(self.targets)} targets.")

        if len(args.optional_meta)>0:
            self.optional_meta = np.array(self.demo[args.optional_meta]).astype('float32')

        else:
            self.optional_meta = np.array([])


    def __getitem__(self, index):
        # Get target as float tensor
        target = torch.tensor([self.demo[self.targetname].iloc[index]], dtype=torch.float32)

        path = self.image_paths[index]
        

        # Load images as torchio images
        image = tio.ScalarImage(path)
        image = self.resize(image)
        image_tensor = image.data

        if len(self.optional_meta) > 0:
            meta = torch.tensor(self.optional_meta[index], dtype=torch.float32)
            return [image_tensor, meta, target]

        else:
            return [image_tensor, target]

        
    def __len__(self):
        return len(self.image_paths)