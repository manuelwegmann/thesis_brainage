import matplotlib.pyplot as plt
import torch
from loader import loader3D
import argparse
import os

# Dummy argparse-like object for testing
class Args:
    data_directory = '/mimer/NOBACKUP/groups/brainage/data/oasis3'  # Replace with your actual path
    clean = True
    preprocess_cat = True
    image_size = (128, 128, 128)  # Example size, adjust to your setup
    target_name = 'age'  # Replace with your actual target
    optional_meta = ['education']  # Adjust as needed

# Path to save the output images
output_dir = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results'

def plot_and_save_image_slices(image, title_prefix, save_prefix):
    # Assume image shape is (1, D, H, W)
    image = image.squeeze()  # Remove channel dimension if present
    d, h, w = image.shape
    mid_slices = [d // 2, h // 2, w // 2]
    planes = ['Axial', 'Coronal', 'Sagittal']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (plane, slice_idx) in enumerate(zip(planes, mid_slices)):
        if plane == 'Axial':
            slice_img = image[slice_idx, :, :]
        elif plane == 'Coronal':
            slice_img = image[:, slice_idx, :]
        elif plane == 'Sagittal':
            slice_img = image[:, :, slice_idx]

        axes[i].imshow(slice_img.T, cmap='gray', origin='lower')
        axes[i].set_title(f'{title_prefix} - {plane}')
        axes[i].axis('off')

        # Save each plane as separate image
        out_path = os.path.join(output_dir, f"{save_prefix}_{plane.lower()}.png")
        plt.imsave(out_path, slice_img.T, cmap='gray', origin='lower')

    # Save combined figure
    fig_path = os.path.join(output_dir, f"{save_prefix}_combined.png")
    plt.savefig(fig_path)
    plt.close(fig)

if __name__ == '__main__':
    args = Args()
    dataset = loader3D(args)

    print(f"Loaded dataset with {len(dataset)} samples.")
    
    for i in range(2):  # Visualize and save first 2 samples
        print(f"\nSample {i}")
        sample = dataset[i]
        if len(sample) == 4:
            image1, image2, meta, target = sample
            print(f"Meta: {meta}, Target: {target}")
        else:
            image1, image2, target = sample
            print(f"Target: {target}")

        plot_and_save_image_slices(image1, f'Sample {i} - Image1', f'sample{i}_image1')
        plot_and_save_image_slices(image2, f'Sample {i} - Image2', f'sample{i}_image2')

    print(f"\nSaved slice images to: {os.path.abspath(output_dir)}")