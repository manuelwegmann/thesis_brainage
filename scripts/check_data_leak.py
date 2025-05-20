import os
import numpy as np
import pandas as pd
from new_loader import load_participants, loader3D
from sklearn.model_selection import train_test_split



class Args:
    data_directory = '/mimer/NOBACKUP/groups/brainage/data/oasis3'  # Replace with your actual path
    clean = True
    preprocess_cat = True
    image_size = (128, 128, 128)  # Example size, adjust to your setup
    target_name = 'duration'  # Replace with your actual target
    optional_meta = ['sex_F', 'sex_M', 'sex_U', 'age']  # Adjust as needed

args = Args()

df = load_participants()
df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
data_train = loader3D(args, df_train)
data_val = loader3D(args, df_val)
data_test = loader3D(args, df_test)

d1 = data_train.demo
d2 = data_val.demo
d3 = data_test.demo

ids1 = d1['participant_id'].unique()
ids2 = d2['participant_id'].unique()
ids3 = d3['participant_id'].unique()

overlap_1_2 = np.intersect1d(ids1, ids2)
overlap_1_3 = np.intersect1d(ids1, ids3)
overlap_2_3 = np.intersect1d(ids2, ids3)

any_overlap = overlap_1_2.size > 0 or overlap_1_3.size > 0 or overlap_2_3.size > 0

print("Any overlap:", any_overlap)

print("We are now going to check if there is any data leackage in our actual training files.")

path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/run_after_changes'
df1 = pd.read_csv(os.path.join(path,'train_dataset.csv'), sep=',')
df2 = pd.read_csv(os.path.join(path,'val_dataset.csv'), sep=',')
df3 = pd.read_csv(os.path.join(path,'test_dataset.csv'), sep=',')

ids1 = df1['participant_id'].unique()
ids2 = df2['participant_id'].unique()
ids3 = df3['participant_id'].unique()

ex1 = df1['sex'].unique()
ex2 = df2['sex'].unique()
intersect = np.intersect1d(ex1,ex2)
overlap = intersect.size > 0
print("The method of checking works:", overlap)

overlap_1_2 = np.intersect1d(ids1, ids2)
overlap_1_3 = np.intersect1d(ids1, ids3)
overlap_2_3 = np.intersect1d(ids2, ids3)

any_overlap = overlap_1_2.size > 0 or overlap_1_3.size > 0 or overlap_2_3.size > 0

print("Any overlap:", any_overlap)

