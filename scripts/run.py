from .loader import loader3D
from .model import LILAC

from utils import *
import torch
import numpy as np
import os
import time
import datetime
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="name of job")

    #data preprocessing arguments
    parser.add_argument('--clean', default=True, type=bool, help="whether to clean data from CI and single scan participants")
    parser.add_argument('--preprocess_cat', default=True, type=bool, help="whether to preprocess categorical data")
    parser.add_argument('--image_size', default=[128,128,128], type=list, help="size of the image")
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--test_size', default=0.2, type=float, help="test size for splitting the data")
    parser.add_argument('--seed', default=42, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', default=['age', 'sex_F', 'sex_M'], type=list, help="list of optional meta to be used in the model")

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

    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")


    args = parser.parse_args()

    return args


def split(opt):
    """
    Splits the data into training and testing sets.
    """
    full_dataset = loader3D(opt)
    test_size = int(opt.test_size * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(opt.seed))
    loader_train = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False)

    return loader_train, loader_test


def train(opt, loader_train):
    """
    Trains the model.
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model, Loss, Optimizer
    model = LILAC(opt).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    dataloader = loader_train
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(opt.output_directory, 'logs'))

    # Training loop
    for epoch in range(opt.epoch, opt.max_epoch):
        model.train()
        total_loss = 0

        for batch in dataloader:
            # Unpack batch
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

        #Log the average loss after each epoch
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    # Save the trained model
    save_path = os.path.join(opt.output_directory, "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

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



