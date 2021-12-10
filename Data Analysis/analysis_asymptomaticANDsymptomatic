import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read csv file and import as dataframe
filename_csv = 'clean_data_mx.csv'
# filename_csv = 'clean_data_mx_covidresults.csv'
df = pd.read_csv(filename_csv)
df.drop(columns='Unnamed: 0', axis=1, inplace=True)

# Print header
print(df.head())
# df = df.rename(columns={'tos': 'Cough', 'fiebre': 'Fever', 'odinogia': 'Sore Throat', 'disnea': 'Shortness of Breath',
# 'cefalea': 'Headache', 'edad': 'Age', 'sexo': 'Gender'})

label = {'Test\nResult', 'Cough', 'Fever', 'Sore\nthroat', 'Shortness\nof Breath', 'Headache', 'Age\nOver 60'}
correlations = df.corr(method='pearson')
corr_map = sns.heatmap(correlations, annot=True, xticklabels=[], square=True, fmt='.2f')
corr_map.set_yticklabels(['Test\nResult', 'Cough', 'Fever', 'Sore\nthroat', 'Shortness\nof Breath', 'Headache', 'Age\nOver 60'])
plt.show()
# plt.close()

# Jaccard index computation
# Declare zeros matrix
jac_index = np.zeros((7, 7), dtype=float)

# Compute Jaccard index between all variables
for i in range(7):
    label1 = df.iloc[:, i]
    for j in range(7):
        label2 = df.iloc[:, j]
        jac_index[i][j] = jaccard_score(label1, label2)

# Print Results
print(jac_index)
j_map = sns.heatmap(jac_index, annot=True, square=True, fmt='.2f', xticklabels=[])
# j_map.set_yticklabels(['Test\nResult', 'Cough', 'Fever', 'Sore\nthroat', 'Shortness\nof Breath', 'Headache', 'Age\nOver 60'])
plt.show()
