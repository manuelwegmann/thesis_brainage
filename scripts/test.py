from loader import loader3D
from prep_data import basic_data_load
import pandas as pd

class Args:
    data_directory = '/mimer/NOBACKUP/groups/brainage/data/oasis3'  # Replace with your actual path
    clean = True
    preprocess_cat = True
    image_size = (128, 128, 128)  # Example size, adjust to your setup
    target_name = 'age'  # Replace with your actual target
    optional_meta = ['education']  # Adjust as needed

args = Args()
dataset = loader3D(args)
print(f"Loaded dataset with {len(dataset)} samples.")

a = basic_data_load(drop=False)

# Filter rows with at least one missing value
missing_rows = a[a.isnull().any(axis=1)]

# Print the rows with missing values
print("Rows with missing values:")
print(missing_rows)


