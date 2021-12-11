import numpy as np
import pandas as pd

filename_csv = 'corona_tested_individuals_ver_00192.csv'
df=pd.read_csv(filename_csv) #, delimiter=','
# source of csv:
# https://data.gov.il/dataset/covid-19/resource/d337959a-020a-4ed3-84f7-fca182292308/download/corona_tested_individuals_ver_00192.csv

# drops
df.drop(['test_indication'], axis=1, inplace=True) # drop this column
df.drop(['test_date'], axis=1, inplace=True) # drop this column
df.replace('אחר', np.NaN, inplace=True) # replace hebrew 'other' with NaN
df.dropna(axis=0, inplace=True) # drops rows with NaN values

# replace in corona_result label
df.replace('חיובי', 1, inplace=True) # replace hebrew 'positive' with 1
# drop all negative covid inputs
df.replace('שלילי', np.NaN, inplace=True) # replace hebrew 'other' with NaN
df.dropna(axis=0, inplace=True) # drops rows with NaN values
df.drop(['corona_result'], axis=1, inplace=True) # drop this column

# replace in age_60_and_above feature
df.replace('Yes', 1, inplace=True) # replace Yes with 1
df.replace('No', 0, inplace=True) # replace No with 0

# replace in gender feature
df.replace('נקבה', 1, inplace=True) # replace hebrew 'female' with 1
df.replace('זכר', 0, inplace=True) # replace hebrew 'male' with 0

df.to_csv('Israel_dataset_clean.csv')
