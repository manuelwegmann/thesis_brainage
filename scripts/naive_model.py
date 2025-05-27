from prep_data import full_data_load
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = full_data_load(clean=True, drop=True)
df1, df2 = train_test_split(df, test_size=0.2, random_state=42)
naive_pred = np.mean(df1['duration'])
loss_train = np.mean((df1['duration'] - naive_pred)**2)
loss_test = np.mean((df2['duration'] - naive_pred)**2)
print(f"Naive model (average) prediction: {naive_pred}") #6.24
print(f"Naive model (average) loss (train): {loss_train}") #13.18
print(f"Naive model (average) loss (test): {loss_test}") #11.82
loss0_train = np.mean((df1['duration'])**2)
loss0_test = np.mean((df2['duration'])**2)
print(f"Naive model (zero) loss (train): {loss0_train}") #52.09
print(f"Naive model (zero) loss (test): {loss0_test}") #43.74


from new_loader import loader3D, load_participants

print("We now get a naive prediction from the new loader.")


class Args:
    data_directory = '/mimer/NOBACKUP/groups/brainage/data/oasis3'  # Replace with your actual path
    clean = True
    preprocess_cat = True
    image_size = (128, 128, 128)  # Example size, adjust to your setup
    target_name = 'duration'  # Replace with your actual target
    optional_meta = ['sex_F', 'sex_M', 'sex_U', 'age']  # Adjust as needed

args = Args()

df = load_participants()
df1, df2 = train_test_split(df, test_size=0.2, random_state=42)
data_train = loader3D(args, df1)
data_test = loader3D(args, df2)
d1 = data_train.demo
d2 = data_test.demo
naive_pred = np.mean(d1['duration'])
loss_train = np.mean((d1['duration'] - naive_pred)**2)
loss_test = np.mean((d2['duration'] - naive_pred)**2)
print(f"Naive model (average) prediction: {naive_pred}") #4.71
print(f"Naive model (average) loss (train): {loss_train}") #9.20
print(f"Naive model (average) loss (test): {loss_test}") #9.12
print(f"Naive model (average) MAE (test):", f"{np.mean(np.abs(d2['duration'] - naive_pred))}") #2.47
MAE = np.mean(np.abs(d2['duration']-naive_pred))
print(f"MAE: {MAE}")
loss0_train = np.mean((d1['duration'])**2)
loss0_test = np.mean((d2['duration'])**2)
print(f"Naive model (zero) loss (train): {loss0_train}") #31.38
print(f"Naive model (zero) loss (test): {loss0_test}") #32.24
print(f"Naive model (zero) MAE (test): {np.mean(np.abs(d2['duration']))}") #4.81
