import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = '/mimer/NOBACKUP/groups/brainage/thesis_brainage/results/5-fold-cc'
results = pd.read_csv(os.path.join(path,'predicted_values.csv'))

#plot
plt.figure(figsize=(6, 6))
plt.scatter(results['Target'], results['Prediction'], alpha=0.5, label='Data points')
min_val = min(results['Target'].min(), results['Prediction'].min())
max_val = max(results['Target'].max(), results['Prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.title('Target vs Prediction (CI participants)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()    
plot_path = os.path.join(path, 'pred_target.png')
plt.savefig(plot_path)
plt.close()



