import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_file', default='/mimer/NOBACKUP/groups/brainage/thesis_brainage/results', type=str, help="directory of the results to be evaluated")

    args = parser.parse_args()

    return args


def decide_type_of_result(filepath):
    if 'train' in filepath.lower():
        name = 'train'
    elif 'val' in filepath.lower():
        name = 'val'
    elif 'test' in filepath.lower():
        name = 'test'
    else:
        print("Error in naming of file.")
        name = 'error'
    return name



def correlation_analysis(predictions, targets):
    r, p = pearsonr(predictions, targets)
    return r, p


def scatter_plot(predictions, targets, save_path, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(predictions, targets, alpha=0.5, label='Data points')
    min_val = min(predictions.min(), targets.min())
    max_val = max(predictions.max(), targets.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (opti for healthy)')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.title('Target vs Prediction (' + name + ')')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_path, name + '_scatter_plot.png')
    plt.savefig(plot_path)
    plt.close()



if __name__ == "__main__":

    args = parse_args()

    name = decide_type_of_result(args.results_file)
    print(f"We are looking at {name} results.")

    results = pd.read_csv(args.results_file)
    predictions = results.iloc[:, 0].values
    targets = results.iloc[:,1].values
    
    #prepare data directory for saving outputs.
    dir = os.path.dirname(args.results_file)
    folder_name = "results_" + name
    save_path = os.path.join(dir,folder_name)
    os.makedirs(save_path, exist_ok=True)

    #save results in dataframe
    df = pd.DataFrame()

    #Correlation analysis
    r,p = pearsonr(predictions, targets)
    df['PCC'] = [r]
    df['p-value'] = [p]

    #scatterplot
    scatter_plot(predictions, targets, save_path, name)

    #save results stored in dataframe
    df.to_csv(os.path.join(save_path, f'{name}_evaluation.csv'), index=False)






