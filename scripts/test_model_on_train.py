from new_loader import loader3D, load_participants
from LILAC import LILAC

import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")

    #data preprocessing arguments
    parser.add_argument('--clean', default=True, type=bool, help="whether to clean data from CI and single scan participants")
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")
    parser.add_argument('--val_size', default=0.2, type=float, help="validation size for splitting the data")
    parser.add_argument('--test_size', default=0.2, type=float, help="test size for splitting the data")
    parser.add_argument('--seed', default=15, type=int)

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
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


def test(opt, model, test_dataset):
    """
    Tests the model.
    Input:
        opt: options from the command line
        model: trained model
        test_dataset: dataframe with the testing set
    Output:
        prints out the test loss and MAE
        saves the test results to a CSV file
    """
    print("We are in test.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader_test = DataLoader(loader3D(opt, test_dataset), batch_size=opt.batchsize, shuffle=False)

    criterion = nn.MSELoss()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    # list store losses (for plot later)
    test_losses = []

    print("We are in test")

    with torch.no_grad():
        for batch in dataloader_test:
            if len(batch) == 3:
                x1, x2, target = batch
                meta = None
            else:
                x1, x2, meta, target = batch
                meta = meta.to(device)

            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)

            print(f"x1 shape: {x1.shape}")
            print(f"x2 shape: {x2.shape}")
            print(f"meta shape: {meta.shape}")
            print(f"target shape: {target.shape}")

            output = model(x1, x2, meta)
            print(f"output shape: {output.shape}")
            loss = criterion(output, target)
            total_loss += loss.item()

            all_targets.append(target.cpu().numpy())
            all_preds.append(output.cpu().numpy())

    avg_loss = total_loss / len(dataloader_test)
    test_losses.append(avg_loss)
    print(f"Test Loss (MSE): {avg_loss:.4f}")

    # Optionally, save predictions and targets to CSV
    targets = np.concatenate(all_targets, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    results_df = pd.DataFrame({
        "Target": targets.flatten(),
        "Prediction": preds.flatten()
    })
    results_path = os.path.join(opt.output_directory, 'run_after_changes', "new_test_predicted_values.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")

    mae = np.mean(np.abs(preds - targets))
    print(f"Test MAE: {mae:.4f}")

if __name__ == "__main__":
    print("We are in main.")

    # Parse command line arguments
    opt = parse_args()

    # Create output directory
    output_dir = os.path.join(opt.output_directory, 'run_after_changes')

    test_dataset = pd.read_csv(os.path.join(output_dir, 'test_dataset.csv'))
    print("We have loaded the test participants")

    # Model, Loss, Optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LILAC(opt).to(device)
    checkpoint = torch.load('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/run_after_changes/best_model.pt', map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #test model
    test(opt, model, test_dataset)

