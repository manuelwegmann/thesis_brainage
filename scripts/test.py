from loader import loader3D

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