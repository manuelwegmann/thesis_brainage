import torch
import torch.nn as nn
import torch.optim as optim
from .LILAC import LILAC  # assumes the file contains the class LILAC
from .LILAC import get_backbone  # assumes the file contains the function get_backbone
from .loader import loader3D  # assumes this returns a DataLoader or dataset
from torch.utils.data import DataLoader

class Args:
    image_size = [16, 16, 16]  # Size of the image
    target_name = 'duration'  # Target variable
    data_directory = '/mimer/NOBACKUP/groups/brainage/data/oasis3'  # Data location
    optional_meta = ['mr_sessions', 'age', 'sex_F', 'sex_M']  # Optional metadata (fixed as string, no tuple)
    clean = True  # Whether to clean data
    n_of_blocks = 3  # Integer value (no tuple)
    channels = 1  # Number of input channels
    initial_channel = 4  # Initial channel size
    dropout = 0  # Dropout rate (correct spelling)
    pooling = nn.AvgPool3d  # Pooling function
    image_channel = 1  # Number of channels in the input image
    preprocess_cat = True  # Preprocess categorical data (fixed as boolean)

# Setup args as an instance of Args
args = Args()

get_backbone(args)  # Call the function to get the backbone and linear layers

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model, Loss, Optimizer
model = LILAC(args).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Load your data
full_dataset = loader3D(args)
subset_dataset = torch.utils.data.Subset(full_dataset, list(range(10)))
dataloader = DataLoader(subset_dataset, batch_size=2, shuffle=False)

# Training loop
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Unpack batch — update based on your loader structure
        if len(batch) == 3:
            x1, x2, target = batch
            meta = None
        else:
            x1, x2, meta, target = batch
            meta = meta.float().to(device)

        # Move tensors to device
        x1 = torch.tensor(x1).float().to(device)
        x2 = torch.tensor(x2).float().to(device)
        target = torch.tensor(target).float().unsqueeze(1).to(device)

        # Forward pass
        output = model(x1, x2, meta)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calculate and print average loss
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {avg_loss:.4f}")