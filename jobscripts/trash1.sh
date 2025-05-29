#!/bin/env bash

#SBATCH -A NAISS2025-22-353     # project ID found via "projinfo"
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 47:59:59          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1    # choosing no. GPUs and their type

# load modules
module load virtualenv/20.26.2-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

source /mimer/NOBACKUP/groups/brainage/thesis_brainage/my_venv/bin/activate

# execute 
cd /mimer/NOBACKUP/groups/brainage/thesis_brainage/scripts

python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_0/train_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_0/val_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_0/test_predicted_values.csv

python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_1/train_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_1/val_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_1/test_predicted_values.csv

python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_2/train_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_2/val_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_2/test_predicted_values.csv

python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_3/train_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_3/val_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_3/test_predicted_values.csv

python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_4/train_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_4/val_predicted_values.csv
python -u evaluation_pipeline.py --results_file /mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cv_wo_age/5-fold-cv_wo_age/fold_4/test_predicted_values.csv