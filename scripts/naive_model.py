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
