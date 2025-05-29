import torch
import torch.nn as nn
from LILAC import LILAC
import json
from argparse import Namespace

# Set paths
path_to_json = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_w_age/run_details.json'
path_to_best_model = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_w_age/5-fold-cv_w_age/fold_4/best_model.pt'

# Load args from JSON
with open(path_to_json, 'r') as f:
    args_dict = json.load(f)
args = Namespace(**args_dict)

# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")

# Initialize model on CPU
model = LILAC(args).to(device)

# Load checkpoint onto CPU
checkpoint = torch.load(path_to_best_model, map_location=device)

# Restore model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Count only trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

model.eval()

# Extract final FC weights
fc_weights = model.linear.weight.data.numpy()[0]  # already on CPU

print(f"Max weight: {max(fc_weights)}")
print(f"Min weight: {min(fc_weights)}")

# Extract meta data weights
meta = args.optional_meta
num_meta = len(meta)
meta_weights = fc_weights[-num_meta:]

# Print weights
for name, weight in zip(meta, meta_weights):
    print(f"{name}: {weight:.4f}")