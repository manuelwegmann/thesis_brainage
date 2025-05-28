import numpy as np
import pandas as pd
import os


#function to load standard dataset description
def load_basic_overview(file_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3/participants.tsv'): #filepath points to participants.tsv file in OASIS-3 folder
    df = pd.read_csv(file_path, sep='\t')
    return df



# For a given participant ID, find their actual age at baseline
def extract_age_at_baseline(participant_id, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    return pd.read_csv(file_path, sep='\t').iloc[0]['age']

# Update the whole dataset with correct ages at baseline
def add_ages(df, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):  # df is the DataFrame, and folder_path points to the OASIS-3 folder
    df['age'] = df['participant_id'].apply(lambda participant_id: extract_age_at_baseline(participant_id, folder_path))
    return df



# For a given participant ID, find CN/CI classification at baseline and final session
def extract_class_at_baseline(participant_id, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    classification = "CN"
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    if pd.read_csv(file_path, sep='\t').iloc[0]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

def extract_class_at_final(participant_id, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    classification = "CN"
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')
    session_file = pd.read_csv(file_path, sep='\t')
    if session_file.iloc[session_file.shape[0] - 1]['cognitiveyly_normal'] == False:
        classification = "CI"
    return classification

# Update the whole dataset with correct classifications at baseline and final session
def add_classification(df, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    df['class_at_baseline'] = df['participant_id'].apply(lambda participant_id: extract_class_at_baseline(participant_id, folder_path))
    df['class_at_final'] = df['participant_id'].apply(lambda participant_id: extract_class_at_final(participant_id, folder_path))
    return df



#extract time between first and last scan for a given participant id
def extract_duration(participant_id, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    file_path = os.path.join(folder_path, str(participant_id), 'sessions.tsv')  
    sessions_file = pd.read_csv(file_path, sep='\t')
    num_sessions = sessions_file.shape[0]
    baseline = sessions_file.iloc[0]['days_from_baseline']
    final = sessions_file.iloc[num_sessions-1]['days_from_baseline']
    return (final - baseline) / 365.0  # Convert to years

# Add duration column to the DataFrame
def add_duration(df, folder_path='/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    df['duration'] = df['participant_id'].apply(lambda participant_id: extract_duration(participant_id, folder_path))
    return df



#check the whole dataset if any folders with data are missing
def check_folders_exist(df, folder_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3'):
    checked_df = df
    for participant_id in checked_df['participant_id']:
        if os.path.exists(os.path.join(folder_path, str(participant_id))) == False:
            print(f"Warning: Folder for participant {participant_id} does not exist and will be deleted.")
            index_to_drop = df[df['participant_id'] == participant_id].index
            checked_df = checked_df.drop(index_to_drop)
    return checked_df



#check if there are any participants with only 1 scan
def exclude_single_scan_participants(df):
    filtered_df = df[df['mr_sessions'] >= 2]
    excluded_df = df[df['mr_sessions'] < 2]
    #print(f'There are {filtered_df.shape[0]} subjects with at least 2 scans.')
    #print(f'There are {excluded_df.shape[0]} subjects with only 1 scan.')
    return filtered_df

#check if there are any participants with CI at baseline or final session
def exclude_CI_participants(df):
    filtered_df = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CN')]
    return filtered_df



def full_data_load(fp_oasis = '/mimer/NOBACKUP/groups/brainage/data/oasis3', clean = False, preprocess_cat = False, drop = True):
    print("Preprocessing of categorcial data is set to: ", preprocess_cat)
    fp_participants = os.path.join(fp_oasis, 'participants.tsv')
    df = load_basic_overview(file_path = fp_participants) #load information for all subjects
    df = check_folders_exist(df=df, folder_path=fp_oasis) #check if subjects in participants.tsv have data folder
    df = add_ages(df=df, folder_path=fp_oasis) #add ages at baseline #needs some fixing
    df = add_duration(df) #add duration between first and last scan #needs some fixing
    df = add_classification(df) #add classification at baseline and final session #needs some fixing
    if clean == True: #clean dataset such that only participants with at least 2 scans and no CI are kept
        df = exclude_single_scan_participants(df)
        df = exclude_CI_participants(df)
    if preprocess_cat == True: # Preprocess 'sex' column (one-hot encoding)
        sex_onehot = pd.get_dummies(df['sex'], prefix='sex').astype(float)
        df = pd.concat([df.drop(columns=['sex']), sex_onehot], axis=1)
    if drop == True:
        df = df.dropna()
    print(df.head())

    return df


def basic_data_load(fp_oasis = '/mimer/NOBACKUP/groups/brainage/data/oasis3', preprocess_cat = False, drop = True):
    fp_participants = os.path.join(fp_oasis, 'participants.tsv')
    df = load_basic_overview(file_path = fp_participants) #load information for all subjects
    df = add_ages(df=df, folder_path=fp_oasis) #add ages at baseline
    df = add_classification(df) #add classification at baseline and final session
    
    df = exclude_single_scan_participants(df) #clean dataset such that only participants with at least 2 scans and no CI are kept
    df = exclude_CI_participants(df) #exclude CI participants

    # Preprocess 'sex' column (one-hot encoding)
    print("Categorical gender data is encoded  with 0/1")
    sex_onehot = pd.get_dummies(df['sex'], prefix='sex').astype(float)
    df = pd.concat([df.drop(columns=['sex']), sex_onehot], axis=1)

    #drop any rows with NaN values
    if drop == True:
        print("Dropping rows with NaN values")
        df = df.dropna()
    print(df.head())

    return df



#split dataset into female and male
def split_by_gender(df):
    df_male = df[df['sex'] == 'M']
    df_female = df[df['sex'] == 'F']
    return df_male, df_female

#split dataset into CN and CI
def split_by_class(df):
    df_CN = df[df['class_at_baseline'] == "CN"]
    df_CI = df[df['class_at_baseline'] == "CI"]
    return df_CN, df_CI
