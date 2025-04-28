from .proper_loader import loader3D
import torchio as tio

# Assuming args is an object with the necessary parameters
class Args:
    def __init__(self, image_size, target_name, data_directory, optional_meta):
        self.image_size = image_size
        self.target_name = target_name
        self.data_directory = data_directory
        self.optional_meta = optional_meta

# Setup args as an instance of Args
args = Args(
    image_size=(128, 128, 128),
    target_name='duration',
    data_directory='/mimer/NOBACKUP/groups/brainage/data/oasis3',
    optional_meta='age'
)

# Initialize the dataset
dataset = loader3D(args)

import os

test_path = '/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep/sub-OAS30001/ses-d0129/sub-OAS30001_ses-d0129_space-MNI152NLin2009cAsym_desc-brain_T1w.nii.gz'

print('Exists:', os.path.exists(test_path))
print('Can read:', os.access(test_path, os.R_OK))
print('List directory:')
print(os.listdir('/mimer/NOBACKUP/groups/brainage/data/oasis3/derivatives/mriprep/sub-OAS30001/ses-d0129/'))
print("We now test if we can load the image manually:")
test_image = tio.ScalarImage(test_path)
# Check if the image is loaded correctly
print("Image shape:", test_image.shape)

import matplotlib.pyplot as plt

# Load the first sample
sample = dataset[0]

# Unpack based on whether metadata is available
if len(sample) == 4:
    image1, image2, meta, duration = sample
    print(f"Age: {meta}")  # age is optional_meta
else:
    image1, image2, duration = sample
    print("No metadata available.")

print(f"Duration (target): {duration}")

# Choose a central axial slice (axis 2)
central_slice_idx = image1.shape[2] // 2
slice1 = image1[:, :, central_slice_idx]
slice2 = image2[:, :, central_slice_idx]

# Plot the slices side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(slice1.T, cmap="gray", origin="lower")
axs[0].set_title("Image 1 (First Session)")
axs[1].imshow(slice2.T, cmap="gray", origin="lower")
axs[1].set_title("Image 2 (Last Session)")
plt.suptitle("Central Axial Slice of MRI Pair")
plt.tight_layout()
plt.show()
plt.savefig('results/first_slices.png')


# Get the file paths manually from the dataset
path1 = '/mimer/NOBACKUP/groups/brainage/data/oasis3/sub-OAS30001/ses-d0129/anat/sub-OAS30001_ses-d0129_run-01_T1w.nii.gz'
path2 = '/mimer/NOBACKUP/groups/brainage/data/oasis3/sub-OAS30001/ses-d4467/anat/sub-OAS30001_ses-d4467_T1w.nii.gz'

# Load images directly from file paths
manual_image1 = tio.ScalarImage(path1)
manual_image2 = tio.ScalarImage(path2)

# Apply same resizing as dataset
resize = tio.transforms.Resize(dataset.image_size)
manual_image1 = resize(manual_image1)
manual_image2 = resize(manual_image2)

# Convert to numpy
manual_image1_np = manual_image1.numpy().astype("float")
manual_image2_np = manual_image2.numpy().astype("float")

# Extract central slice
central_slice_idx = manual_image1_np.shape[2] // 2
manual_slice1 = manual_image1_np[:, :, central_slice_idx]
manual_slice2 = manual_image2_np[:, :, central_slice_idx]

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(manual_slice1.T, cmap="gray", origin="lower")
axs[0].set_title("Manual Image 1 (First Session)")
axs[1].imshow(manual_slice2.T, cmap="gray", origin="lower")
axs[1].set_title("Manual Image 2 (Last Session)")
plt.suptitle("Central Axial Slice from Manually Loaded Pair")
plt.tight_layout()
plt.show()
plt.savefig("results/manual_first_slices.png")