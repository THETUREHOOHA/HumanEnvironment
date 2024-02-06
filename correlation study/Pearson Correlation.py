#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr

# Read the CSV file
df = pd.read_csv("all_result.csv", encoding='utf-8')

# Calculate Pearson correlation coefficient and p-values
pearsoncorr, p_values = df.corr(method='pearson'), df.corr(method=lambda x, y: pearsonr(x, y)[1])

# Create a figure and heatmap
plt.figure(figsize=(10, 8))
sb.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=False,
            linewidth=0.1,
            fmt=".2f")  # Format correlation values as 2 decimal places

# Add p-values as annotations
for i in range(pearsoncorr.shape[0]):
    for j in range(pearsoncorr.shape[1]):
        if i != j:
            plt.text(j + 0.5, i + 0.5, f"p={p_values.iloc[i, j]:.4f}",
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='black',
                     fontsize=8)

# Set plot title and display the plot
plt.title('Perception Pearson Correlation Coefficient')
plt.show()


# In[ ]:




