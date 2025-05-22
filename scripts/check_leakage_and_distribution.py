import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from new_loader import loader3D

class Args:
    data_directory = '/mimer/NOBACKUP/groups/brainage/data/oasis3'  # Replace with your actual path
    clean = True
    preprocess_cat = True
    image_size = (128, 128, 128)  # Example size, adjust to your setup
    target_name = 'duration'  # Replace with your actual target
    optional_meta = ['sex_F', 'sex_M', 'sex_U', 'age']  # Adjust as needed

args = Args()

path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/new_seed'
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

print("We now look at the age and duration histograms for each dataset.")

data1 = loader3D(args,df1).demo
data2 = loader3D(args,df2).demo
data3 = loader3D(args,df3).demo

age1 = data1['age']
age2 = data2['age']
age3 = data3['age']

duration1 = data1['duration']
duration2 = data2['duration']
duration3 = data3['duration']


def plot_age_histograms(d1, d2, d3, save_path, num_bins=30):
    plt.figure(figsize=(10, 6))

    sns.histplot(d1, bins=num_bins, color='blue', label='Train', alpha=0.5)
    sns.histplot(d2, bins=num_bins, color='red', label='Val', alpha=0.5)
    sns.histplot(d3, bins=num_bins, color='green', label='Test', alpha=0.5)

    plt.xlabel('Age at Baseline (years)')
    plt.ylabel('Frequency')
    plt.title('Age Histogram By Set')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.savefig(os.path.join(save_path, 'age_hist_by_age.png'))

def plot_duration_histograms(d1, d2, d3, save_path, num_bins=30):
    plt.figure(figsize=(10, 6))

    sns.histplot(d1, bins=num_bins, color='blue', label='Train', alpha=0.5)
    sns.histplot(d2, bins=num_bins, color='red', label='Val', alpha=0.5)
    sns.histplot(d3, bins=num_bins, color='green', label='Test', alpha=0.5)

    plt.xlabel('Time between Baseline and Final (years)')
    plt.ylabel('Frequency')
    plt.title('Duration Histogram by Set')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    plt.savefig(os.path.join(save_path, 'duration_hist_by_age.png'))


def plot_age_curves(d1, d2, d3, save_path):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(d1, color='blue', label='Train')
    sns.kdeplot(d2, color='red', label='Val')
    sns.kdeplot(d3, color='green', label='Test')

    plt.xlabel('Age at Baseline (years)')
    plt.ylabel('Density')
    plt.title('Age Distribution by Set (Density)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'age_curves_by_set.png'))  # Added .png extension
    plt.show()

def plot_duration_curves(d1, d2, d3, save_path):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(d1, color='blue', label='Train')
    sns.kdeplot(d2, color='red', label='Val')
    sns.kdeplot(d3, color='green', label='Test')

    plt.xlabel('Time between Baseline and Final (years)')
    plt.ylabel('Density')
    plt.title('Duration Distribution by Set')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'duration_curves_by_set.png'))
    plt.show()


plot_age_histograms(age1, age2, age3, save_path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/new_seed')
plot_age_curves(age1, age2, age3, save_path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/new_seed')
plot_duration_histograms(duration1, duration2, duration3, save_path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/new_seed')
plot_duration_curves(duration1, duration2, duration3, save_path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/new_seed')