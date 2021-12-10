import numpy as np
import pandas as pd

# Read csv file and import as dataframe
filename_csv = 'sisver_public.csv'
df = pd.read_csv(filename_csv)  # , delimiter=','

# View summary of dataset
# print(df.head())  # prints first 5 rows
# print(df.info())  # information about this data frame
# print(df.keys())  # prints different values in the dataframe
# print('Size = ', df.shape)  # prints size of dataframe
# print(df.describe())  # prints statistical information about dataframe

# Drop unnecessary information
df.drop(df.iloc[:, 0:7], inplace=True, axis=1)
df.drop(df.iloc[:, 1:7], inplace=True, axis=1)
df.drop(df.iloc[:, 3:5], inplace=True, axis=1)
df.drop(df.iloc[:, 6:15], inplace=True, axis=1)
df.drop(df.iloc[:, 37:46], inplace=True, axis=1)
df.drop(df.iloc[:, 38:42], inplace=True, axis=1)
df.drop(df.iloc[:, 39:53], inplace=True, axis=1)
df.drop(columns=df.columns[-1], inplace=True, axis=1)
print(df.keys())
print('Size = ', df.shape)

# Find label for positive covid-19 results
# d1 = df['resdefin']
# print(d1.unique())
# print(d1.value_counts()) # 1,298,902 negative; 374,730 positive

# Replace column values in COVID test results

# cast the dtype to int this will convert True and False to 1 and 0 respectively
df['resdefin'] = (df['resdefin'] == "SARS-CoV-2").astype(int).replace(0, np.nan)

# Check results
# d1 = df['resdefin']
# print(d1.unique())
# print(d1.value_counts()) # 1,298,902 negative; 374,730 positive

# Check types of labels before replacing
# d2 = df['fiebre']
# print(d2.unique()) # labels are "SE IGNORA" "NO" "SI"

# Replace all "SE IGNORA" with NaN
df.replace('SE IGNORA', np.NaN, inplace=True)

# Replace all SI with 1 and NO with 0
df.replace('SI', 1, inplace=True)
df.replace('NO', 0, inplace=True)

# Get gender labels
# d1 = df['sexo']
# print(d1.unique())
# print(d1.value_counts()) # 1,298,902 negative; 374,730 positive
# Replace gender
df.replace('MASCULINO', '0', inplace=True)
df.replace('FEMENINO', '1', inplace=True)

# Replace age to older than 60 (=1) and younger than 60 (=0)
df['edad'] = (df['edad'] > 60).astype(int)
# df[df.edad > 60] = 1
# df[df.edad < 61] = 0

# Make new dataframe exactly like Israeli dataset
df_clean = df[['resdefin', 'tos', 'fiebre', 'odinogia', 'disnea', 'cefalea', 'edad', 'sexo']]
print(df_clean.head())
df_clean.dropna(axis=0, inplace=True)  # drops rows with NaN values
#df_clean.drop(columns='resdefin', axis=1, inplace=True)
df_clean = df_clean.rename(columns={'resdefin': 'Test Result', 'tos': 'Cough', 'fiebre': 'Fever', 'odinogia': 'Sore Throat', 'disnea': 'Shortness of Breath',
                        'cefalea': 'Headache', 'edad': 'Age', 'sexo': 'Gender'})
print(df_clean.head())
print('Size = ', df_clean.shape)

# Save to new csv file (uncomment one)
#df_clean.to_csv('clean_data_mx_covidresults.csv')
#df_clean.to_csv('clean_data_mx.csv')
