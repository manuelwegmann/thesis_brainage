from new_loader import loader3D, load_participants
from LILAC import LILAC
from prep_data import exclude_single_scan_participants, add_classification, check_folders_exist

import torch
import numpy as np
import os
import json
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory', default='/mimer/NOBACKUP/groups/brainage/data/oasis3', type=str, help="directory of the data (OASIS3)")

    #data preprocessing arguments
    parser.add_argument('--image_size', nargs=3, type=int, default=[128, 128, 128], help='Input image size as three integers (e.g. 128 128 128)')
    parser.add_argument('--image_channel', default=1, type=int, help="number of channels in the input image")

    #target and optional meta data arguments
    parser.add_argument('--target_name', default='duration', type=str, help="name of the target variable")
    parser.add_argument('--optional_meta', nargs='+', default=['age', 'sex_F', 'sex_M'], help="List of optional meta to be used in the model")
    
    #model architecture arguments
    parser.add_argument('--n_of_blocks', default=4, type=int, help="number of blocks in the encoder")
    parser.add_argument('--initial_channel', default=16, type=int, help="initial channel size after first conv")
    parser.add_argument('--kernel_size', default=3, type=int, help="kernel size")
    parser.add_argument('--conv_act', default='leaky_relu', type=str, help="activation function")

    #test arguments
    parser.add_argument('--batchsize', default=16, type=int)

    parser.add_argument('--output_directory', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory path for saving model and outputs")
    parser.add_argument('--run_name', default='test_run', type=str, help="name of the run")


    args = parser.parse_args()

    return args

def load_args_from_json(json_path):
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

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
    results_path = os.path.join(opt.output_directory, opt.run_name, "predicted_values.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Test results saved to {results_path}")

    mae = np.mean(np.abs(preds - targets))
    print(f"Test MAE: {mae:.4f}")

if __name__ == "__main__":
    print("We are in main.")

    # Parse command line arguments
    args = load_args_from_json('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cc/run_details.json')

    """
    Load desired data for testing.
    """
    df = pd.read_csv(os.path.join(args.data_directory, 'participants.tsv'), sep='\t')
    df = check_folders_exist(df)
    df = add_classification(df)
    df = exclude_single_scan_participants(df)
    dfCI = df[(df['class_at_baseline'] == 'CI')]
    dfCNCI = df[(df['class_at_baseline'] == 'CN') & (df['class_at_final'] == 'CI')]
    dfCI = dfCI[['participant_id', 'sex']]
    dfCNCI = dfCNCI[['participant_id', 'sex']]
    

    # Model, Loss, Optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LILAC(args).to(device)
    checkpoint = torch.load('/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cc/5-fold-cc/fold_2/best_model.pt', map_location = device)
    model.load_state_dict(checkpoint['model_state_dict'])

    #test model
    test(args, model, dfCI)

