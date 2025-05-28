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
    if clean: # Exclude participants with CI and those with only one scan
        df = exclude_CI_participants(df)
        df = exclude_single_scan_participants(df)
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
    Function to extract all available image pairs of a participant. Encode gender as one-hot encoding.
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
    sessions_file = pd.read_csv(sessions_file_path, sep='\t')
    num_sessions = sessions_file.shape[0]
    #check if the participant has at least 2 sessions
    if num_sessions < 2:
        print(f"Warning: Participant {participant_id} has less than 2 sessions. Skipping.")
        return None
    else:
        scan1_list = []
        scan2_list = []
        time_difference_list = []
        age_list = []
        #extract pairs of sessions
        for i in range(num_sessions-1):
            scan1_id = sessions_file.iloc[i]['session_id']
            scan1_session = sessions_file[sessions_file['session_id'] == scan1_id]
            scan1_time = scan1_session.iloc[0]['days_from_baseline']
            age = scan1_session.iloc[0]['age']
            for j in range(i+1, num_sessions):
                if np.isnan(age): #skip if age is not available
                    break
                scan2_id = sessions_file.iloc[j]['session_id']
                scan1_list.append(scan1_id)
                scan2_list.append(scan2_id)
                scan2_session = sessions_file[sessions_file['session_id'] == scan2_id]
                scan2_time = scan2_session.iloc[0]['days_from_baseline']
                time_difference = (scan2_time - scan1_time)/365
                time_difference_list.append(time_difference)
                age_list.append(age)

        return pd.DataFrame({
            'participant_id': [participant_id] * len(scan1_list),
            'sex_M': [sex_M] * len(scan1_list),
            'sex_F': [sex_F] * len(scan1_list),
            'sex_U': [sex_U] * len(scan1_list),
            'age': age_list,
            'session_id1': scan1_list,
            'session_id2': scan2_list,
            'duration': time_difference_list
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
        self.image_pair_paths = []
        valid_demo_rows = []
        for _, row in self.demo.iterrows():
            participant_id = str(row['participant_id'])
            session1 = str(row['session_id1'])
            session2 = str(row['session_id2'])
            img_dir1 = os.path.join(self.datadir, 'derivatives', 'mriprep', participant_id, session1)
            img_dir2 = os.path.join(self.datadir, 'derivatives', 'mriprep', participant_id, session2)
            pattern1 = os.path.join(img_dir1, '*T1w.nii.gz')
            pattern2 = os.path.join(img_dir2, '*T1w.nii.gz')

            matching_files1 = glob.glob(pattern1)
            matching_files2 = glob.glob(pattern2)

            if not matching_files1 or not matching_files2:
                print(f"Warning: No matching T1w image found for {participant_id} in session(s). Skipping.")
                continue #skip if no matching files are found
            path1 = matching_files1[0]
            path2 = matching_files2[0]
            self.image_pair_paths.append((path1, path2))
            valid_demo_rows.append(row)

        self.demo = pd.DataFrame(valid_demo_rows).reset_index(drop=True)
        self.targets = self.demo[self.targetname].values

        #check length of loaded data
        print(f"Loaded {len(self.image_pair_paths)} image pairs.")
        print(f"Loaded {len(self.demo)} rows of metadata.")
        print(f"Loaded {len(self.targets)} targets.")

        if len(args.optional_meta)>0:
            self.optional_meta = np.array(self.demo[args.optional_meta]).astype('float32')

        else:
            self.optional_meta = np.array([])


    def __getitem__(self, index):
        # Get target as float tensor
        target = torch.tensor([self.demo[self.targetname].iloc[index]], dtype=torch.float32)

        path1, path2 = self.image_pair_paths[index]
        

        # Load images as torchio images
        image1 = tio.ScalarImage(path1)
        image2 = tio.ScalarImage(path2)
        image1 = self.resize(image1)
        image2 = self.resize(image2)
        image1_tensor = image1.data
        image2_tensor = image2.data

        if len(self.optional_meta) > 0:
            meta = torch.tensor(self.optional_meta[index], dtype=torch.float32)
            return [image1_tensor, image2_tensor, meta, target]

        else:
            return [image1_tensor, image2_tensor, target]

        
    def __len__(self):
        return len(self.image_pair_paths)