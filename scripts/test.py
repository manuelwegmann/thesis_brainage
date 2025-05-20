import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/run_after_changes'
train_results = pd.read_csv(os.path.join(path,'train_predicted_values.csv'))
val_results = pd.read_csv(os.path.join(path,'val_predicted_values.csv'))
test_results = pd.read_csv(os.path.join(path,'new_test_predicted_values.csv'))

#Train plot
plt.figure(figsize=(6, 6))
plt.scatter(train_results['Target (Train)'], train_results['Prediction (Train)'], alpha=0.5, label='Data points')
min_val = min(train_results['Target (Train)'].min(), train_results['Prediction (Train)'].min())
max_val = max(train_results['Target (Train)'].max(), train_results['Prediction (Train)'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Target vs Prediction (Train)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()    
plot_path = os.path.join(path, 'train_pred_target.png')
plt.savefig(plot_path)
plt.close()

#Val plot
plt.figure(figsize=(6, 6))
plt.scatter(val_results['Target (Val)'], val_results['Prediction (Val)'], alpha=0.5, label='Data points')
min_val = min(val_results['Target (Val)'].min(), val_results['Prediction (Val)'].min())
max_val = max(val_results['Target (Val)'].max(), val_results['Prediction (Val)'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Target vs Prediction (Val)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()    
plot_path = os.path.join(path, 'val_pred_target.png')
plt.savefig(plot_path)
plt.close()

#Test plot
plt.figure(figsize=(6, 6))
plt.scatter(test_results['Target'], test_results['Prediction'], alpha=0.5, label='Data points')
min_val = min(test_results['Target'].min(), test_results['Prediction'].min())
max_val = max(test_results['Target'].max(), test_results['Prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Target vs Prediction (Test)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()    
plot_path = os.path.join(path, 'test_pred_target.png')
plt.savefig(plot_path)
plt.close()


alt_test_results = pd.read_csv(os.path.join(path,'test_predicted_values.csv'))
original_mse = np.mean((test_results['Prediction'] - test_results['Target'])**2)
alt_mse = np.mean((alt_test_results['Prediction'] - alt_test_results['Target'])**2)
print("Test MSE after rerun: ", original_mse)
print("Test MSE before rerun: ", alt_mse)


