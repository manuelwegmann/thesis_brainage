from loader import loader3D
from LILAC import LILAC

import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="name of job")

    #data preprocessing arguments
    parser.add_argument('--clean', default=True, type=bool, help="whether to clean data from CI and single scan participants")
    parser.add_argument('--preprocess_cat', default=True, type=bool, help="whether to preprocess categorical data")
    #parser.add_argument('--image_size', default=[128,128,128], type=list, help="size of the image")
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--val_size', default=0.2, type=float, help="validation size for splitting the data")
    parser.add_argument('--test_size', default=0.2, type=float, help="test size for splitting the data")
    parser.add_argument('--seed', default=15, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    #parser.add_argument('--optional_meta', default=['age', 'sex_F', 'sex_M'], type=list, help="list of optional meta to be used in the model")
    parser.add_argument('--optional_meta', nargs='+', default=['age', 'sex_F', 'sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")
    parser.add_argument('--conv_act', default='leaky_relu', type=str, help="activation function")
    parser.add_argument('--pooling', default=nn.AvgPool3d, type=nn.Module, help="pooling function")

    #training arguments
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=300, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--save_epoch_num', default=1, type=int, help="validate and save every N epoch")
    parser.add_argument('--early_stopping_patience', default=10, type=int, help="early stopping patience")

    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")


    args = parser.parse_args()

    return args


def split(opt):
    """
    Splits the data into training, validation and testing sets.
    """
    full_dataset = loader3D(opt)
    val_size = int(opt.val_size * len(full_dataset))
    test_size = int(opt.test_size * len(full_dataset))
    train_size = len(full_dataset) - val_size - test_size
    train_dataset, temp_dataset = random_split(full_dataset, [train_size, val_size + test_size], generator=torch.Generator().manual_seed(opt.seed))
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(opt.seed))
    loader_train = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False)
    loader_test = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False)
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    return loader_train, loader_val, loader_test



def train(opt, loader_train, loader_val):
    """
    Trains the model with early stopping based on validation loss.
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, Loss, Optimizer
    model = LILAC(opt).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Data loaders
    dataloader_train = loader_train
    dataloader_val = loader_val

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(opt.output_directory, 'logs'))

    # Variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    # lists to store losses (for plot later)
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(opt.epoch, opt.max_epoch):
        print("We are in epoch (training): ", epoch)
        model.train()
        total_loss = 0

        for batch in dataloader_train:
            # Unpack batch
            if len(batch) == 3:
                x1, x2, target = batch
                meta = None
            else:
                x1, x2, meta, target = batch
                meta = meta.float().to(device)

            # Move tensors to device
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            target = target.float().unsqueeze(1).to(device)

            # Print shapes and types
            print("x1:", type(x1), x1.shape if hasattr(x1, 'shape') else "No shape")
            print("x2:", type(x2), x2.shape if hasattr(x2, 'shape') else "No shape")
            print("meta:", type(meta), meta.shape if (meta is not None and hasattr(meta, 'shape')) else "None or No shape")
            print("target:", type(target), target.shape if hasattr(target, 'shape') else "No shape")

            # Forward pass
            output = model(x1, x2, meta)
            #if torch.isnan(x1).any() or torch.isnan(x2).any() or torch.isnan(meta).any() or torch.isnan(target).any():
            #    print("NaN detected in inputs")
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log the average training loss
        avg_train_loss = total_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}")

        # Validation phase (to track validation loss)
        model.eval()
        print("We are in epoch (val): ", epoch)
        total_val_loss = 0
        with torch.no_grad():
            for batch in dataloader_val:
                if len(batch) == 3:
                    x1, x2, target = batch
                    meta = None
                else:
                    x1, x2, meta, target = batch
                    meta = meta.float().to(device)

                # Move tensors to device
                x1 = x1.float().to(device)
                x2 = x2.float().to(device)
                target = target.float().unsqueeze(1).to(device)

                # Forward pass
                output = model(x1, x2, meta)
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        print(f"Epoch {epoch}: Avg Val Loss = {avg_val_loss:.4f}")

        # Check if we need to early stop
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # Save best model weights

            best_model_path = os.path.join(opt.output_directory, 'models', 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss
            }, best_model_path)
            
            print("Validation loss improved, saving best model.")

        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epochs.")

       
        # Early stopping check
        if epochs_without_improvement >= opt.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # Load the best model state (optional)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded the best model.")
    
    # Plot and save training/validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(opt.output_directory, 'training', 'loss_plot_trainval.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss plot (Train/Val) saved to: {plot_path}")

    return model


def test(opt, model, loader_test):
    """
    Tests the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    # list store losses (for plot later)
    test_losses = []

    print("We are in test")

    with torch.no_grad():
        for batch in loader_test:
            if len(batch) == 3:
                x1, x2, target = batch
                meta = None
            else:
                x1, x2, meta, target = batch
                meta = meta.float().to(device)

            x1 = torch.tensor(x1).float().to(device)
            x2 = torch.tensor(x2).float().to(device)
            target = torch.tensor(target).float().unsqueeze(1).to(device)

            output = model(x1, x2, meta)
            loss = criterion(output, target)
            total_loss += loss.item()

            all_targets.append(target.cpu().numpy())
            all_preds.append(output.cpu().numpy())

    avg_loss = total_loss / len(loader_test)
    test_losses.append(avg_loss)
    print(f"Test Loss (MSE): {avg_loss:.4f}")

    # Optionally, save predictions and targets to CSV
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    results_df = pd.DataFrame({
        "Target": targets.flatten(),
        "Prediction": preds.flatten()
    })
    results_path = os.path.join(opt.output_directory, "test_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")



if __name__ == "__main__":
    print("We are in main")
    opt = parse_args()
    loader_test, loader_val, loader_train = split(opt)
    trained_model = train(opt, loader_train, loader_val)
    test(opt, trained_model, loader_test)
