import matplotlib.pyplot as plt
from new_loader import loader3D, load_participants
import os
import pandas as pd
import argparse
import torch.nn as nn

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

if __name__ == "__main__":
    args = parse_args()

    path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/run_after_changes'
    train_participants = pd.read_csv(os.path.join(path,'train_dataset.csv'))
    val_participants = pd.read_csv(os.path.join(path,'val_dataset.csv'))
    test_participants = pd.read_csv(os.path.join(path,'test_dataset.csv'))

    full_train = loader3D(args, train_participants)
    full_val = loader3D(args, val_participants)
    full_test = loader3D(args, test_participants)

    for i in range(len(full_train.demo)):
        ses1 = full_train.demo.iloc[i]['session_id1']
        ses2 = full_train.demo.iloc[i]['session_id2']
        id = full_train.demo.iloc[i]['participant_id']
        pattern1 = os.path.join(id,ses1)
        pattern2 = os.path.join(id,ses2)
        a,path1,path2 = full_train[i]
        if pattern1 in path1 and pattern2 in path2:
            print("All good for (train) ", i)
        else:
            print("Warning for ", i)
            break

    for i in range(len(full_val.demo)):
        ses1 = full_val.demo.iloc[i]['session_id1']
        ses2 = full_val.demo.iloc[i]['session_id2']
        id = full_val.demo.iloc[i]['participant_id']
        pattern1 = os.path.join(id,ses1)
        pattern2 = os.path.join(id,ses2)
        a,path1,path2 = full_val[i]
        if pattern1 in path1 and pattern2 in path2:
            print("All good for (val) ", i)
        else:
            print("Warning for ", i)
            break

    for i in range(len(full_test.demo)):
        ses1 = full_test.demo.iloc[i]['session_id1']
        ses2 = full_test.demo.iloc[i]['session_id2']
        id = full_test.demo.iloc[i]['participant_id']
        pattern1 = os.path.join(id,ses1)
        pattern2 = os.path.join(id,ses2)
        a,path1,path2 = full_test[i]
        if pattern1 in path1 and pattern2 in path2:
            print("All good for (test) ", i)
        else:
            print("Warning for ", i)
            break

