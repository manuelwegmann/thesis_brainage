from prep_data import full_data_load
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = full_data_load(clean=True, drop=True)
df1, df2 = train_test_split(df, test_size=0.2, random_state=42)
naive_pred = np.mean(df1['duration'])
loss_test = np.mean((df2['duration'] - naive_pred)**2)
print(f"Naive model prediction: {naive_pred}")
print(f"Naive model loss: {loss_test}")