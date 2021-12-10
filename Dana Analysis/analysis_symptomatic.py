import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import jaccard_score
from sklearn.cluster import KMeans
################################################################################

# load Israel data
filename_csv = 'Israel_dataset_clean.csv'
df_all_IL = pd.read_csv(filename_csv)
df_all_IL.drop(['Unnamed: 0'], axis=1, inplace=True) # drop this column

# load mexico data
filename_csv = 'clean_data_mx.csv'
df_all_MX=pd.read_csv(filename_csv) #, delimiter=','
df_all_MX.drop(['Unnamed: 0'], axis=1, inplace=True) # drop this column

# replace in gender feature
df_all_MX.replace('F', 1, inplace=True) # replace hebrew 'female' with 1
df_all_MX.replace('M', 0, inplace=True) # replace hebrew 'male' with 0

mean_of_columns_IL = df_all_IL.mean()
mean_of_columns_MX = df_all_MX.mean()

################################################################################
# filter out asymptomatic samples IL
df_IL = df_all_IL.loc[((df_all_IL['Cough'] == 1) | (df_all_IL['Fever'] == 1) | (df_all_IL['Sore_throat'] == 1) | (df_all_IL['Shortness_of_breath'] == 1) | (df_all_IL['Head_ache'] == 1)), :]
df_MX = df_all_MX.loc[((df_all_MX['Cough'] == 1) | (df_all_MX['Fever'] == 1) | (df_all_MX['Sore_throat'] == 1) | (df_all_MX['Shortness_of_breath'] == 1) | (df_all_MX['Head_ache'] == 1)), :]

mean_of_columns_symp_IL = df_IL.mean()
mean_of_columns_symp_MX = df_MX.mean()

df_asymp_IL = df_all_IL.loc[((df_all_IL['Cough'] == 0) & (df_all_IL['Fever'] == 0) & (df_all_IL['Sore_throat'] == 0) & (df_all_IL['Shortness_of_breath'] == 0) & (df_all_IL['Head_ache'] == 0)), :]
df_asymp_MX = df_all_MX.loc[((df_all_MX['Cough'] == 0) & (df_all_MX['Fever'] == 0) & (df_all_MX['Sore_throat'] == 0) & (df_all_MX['Shortness_of_breath'] == 0) & (df_all_MX['Head_ache'] == 0)), :]

mean_of_columns_asymp_IL = df_asymp_IL.mean()
mean_of_columns_asymp_MX = df_asymp_MX.mean()

################################################################################
# Pearson correlation between features
correlations_IL=df_IL.corr(method='pearson')
plt.figure(1)
label = ['Cough', 'Fever', ' Sore \nthroat', 'Shortness\nof      \nbreath   ', 'Head\nache', 'Over\n60  ', 'Gender']
sns.heatmap(correlations_IL, annot=True, xticklabels=[], yticklabels=label, fmt='.2f', square=True)
plt.title('Pearson correlation coefficient Israel only')
# plt.tight_layout()
plt.show()

correlations_MX = df_MX.corr(method='pearson')
plt.figure(2)
sns.heatmap(correlations_MX, annot=True, xticklabels=[], yticklabels=label, fmt='.2f', square=True)
plt.title('Pearson correlation coefficient Mexico only')
# plt.tight_layout()
plt.show()

################################################################################
# Jaccard index Israel
size=df_IL.shape
jac_index_IL = np.zeros((size[1], size[1]), dtype=float)
# Compute Jaccard index between all variables
for i in range(size[1]):
    label1 = df_IL.iloc[:, i]
    for j in range(size[1]):
        label2 = df_IL.iloc[:, j]
        jac_index_IL[i][j] = jaccard_score(label1, label2)

# Print Jaccard index Results
sns.heatmap(jac_index_IL, annot=True, xticklabels=[], yticklabels=label, fmt='.2f', square=True)
plt.title('Jaccard Index Israel only')
plt.show()
################################################################################

# Jaccard index Mexico
jac_index_MX = np.zeros((size[1], size[1]), dtype=float)
# Compute Jaccard index between all variables
for i in range(size[1]):
    label1 = df_MX.iloc[:, i]
    for j in range(size[1]):
        label2 = df_MX.iloc[:, j]
        jac_index_MX[i][j] = jaccard_score(label1, label2)

# Print Jaccard index Results
sns.heatmap(jac_index_MX, annot=True, xticklabels=[], yticklabels=label, fmt='.2f', square=True)
plt.title('Jaccard Index Mexico only')
plt.show()
################################################################################
# use Kmeans to cluster
wcss_IL = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_IL)
    wcss_IL.append(kmeans.inertia_)

wcss_MX = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_MX)
    wcss_MX.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss_IL/wcss_IL[0], label='Israel dataset')
plt.plot(range(1, 11), wcss_MX/wcss_MX[0], label='Mexico dataset')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS- Within Cluster Sum of Squares')
plt.legend()
plt.show()
################################################################################
# select k=2 and use for visualization
kmeans = KMeans(2, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(df_IL)
df_IL['kmeans_index']=np.array(kmeans.labels_) # add another column wih the cluster index
# df_IL.to_csv('IL_dataset_clean_with_cluster.csv')
index_zero_IL_K2 = (df_IL['kmeans_index'].values == 0).sum() # count number of rows in each cluster
index_one_IL_K2 = (df_IL['kmeans_index'].values == 1).sum()
# split the sorted df into 2 dataframes
df_IL_sorted = df_IL.sort_values('kmeans_index') # new dataset sorted by the cluster index
df_cluster_zero_IL = df_IL_sorted.iloc[0:index_zero_IL_K2 - 1, :]
df_cluster_one_IL = df_IL_sorted.iloc[index_zero_IL_K2:, :]
# calc number of counts
total_count_full_IL = df_IL.sum()
total_count_zero_IL = df_cluster_zero_IL.sum()
total_count_one_IL = df_cluster_one_IL.sum()

index_zero_partial_IL = total_count_zero_IL / total_count_full_IL
index_one_partial_IL = total_count_one_IL / total_count_full_IL
################################################################################
# create new df for bar plotting
plot_df_IL = pd.DataFrame([['Is Female', 100 * index_zero_partial_IL.Gender, 100 * index_one_partial_IL.Gender], ['Cough', 100 * index_zero_partial_IL.Cough, 100 * index_one_partial_IL.Cough],
                           ['Fever', 100 * index_zero_partial_IL.Fever, 100 * index_one_partial_IL.Fever], ['Sore throat', 100 * index_zero_partial_IL.Sore_throat, 100 * index_one_partial_IL.Sore_throat],
                           ['Shortness of breath', 100 * index_zero_partial_IL.Shortness_of_breath, 100 * index_one_partial_IL.Shortness_of_breath], ['Head ache', 100 * index_zero_partial_IL.Head_ache, 100 * index_one_partial_IL.Head_ache],
                           ['Is 60 and over', 100 * index_zero_partial_IL.Age_60_and_over, 100 * index_one_partial_IL.Age_60_and_over]],
                          columns=['type', 'Cluster 1', 'Cluster 2'])

plot_df_IL.plot(x='type', kind='barh', stacked=False, title='K-Means clustered data, with k=2, IL dataset')
plt.xlabel('Percentage in each cluster', fontsize=12)
plt.xlim([0, 100])
plt.ylabel('')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()
###################################################
# MX dataset
kmeans.fit(df_MX)
df_MX['kmeans_index']=np.array(kmeans.labels_) # add another column wih the cluster index
# df_MX.to_csv('MX_dataset_clean_with_cluster.csv')
index_zero_MX_K2 = (df_MX['kmeans_index'].values == 0).sum() # count number of rows in each cluster
index_one_MX_K2 = (df_MX['kmeans_index'].values == 1).sum()
# split the sorted df into 2 dataframes
df_MX_sorted = df_MX.sort_values('kmeans_index') # new dataset sorted by the cluster index
df_cluster_zero_MX = df_MX_sorted.iloc[0:index_zero_MX_K2 - 1, :]
df_cluster_one_MX = df_MX_sorted.iloc[index_zero_MX_K2:, :]
# calc number of counts
total_count_full_MX = df_MX.sum()
total_count_zero_MX = df_cluster_zero_MX.sum()
total_count_one_MX = df_cluster_one_MX.sum()

index_zero_partia_MX = total_count_zero_MX / total_count_full_MX
index_one_partial_MX = total_count_one_MX / total_count_full_MX
################################################################################
# create new df for bar plotting
plot_df_MX = pd.DataFrame([['Is Female', 100 * index_zero_partia_MX.Gender, 100 * index_one_partial_MX.Gender], ['Cough', 100 * index_zero_partia_MX.Cough, 100 * index_one_partial_MX.Cough],
                           ['Fever', 100 * index_zero_partia_MX.Fever, 100 * index_one_partial_MX.Fever], ['Sore throat', 100 * index_zero_partia_MX.Sore_throat, 100 * index_one_partial_MX.Sore_throat],
                           ['Shortness of breath', 100 * index_zero_partia_MX.Shortness_of_breath, 100 * index_one_partial_MX.Shortness_of_breath], ['Head ache', 100 * index_zero_partia_MX.Head_ache, 100 * index_one_partial_MX.Head_ache],
                           ['Is 60 and over', 100 * index_zero_partia_MX.Age_60_and_over, 100 * index_one_partial_MX.Age_60_and_over]],
                          columns=['type', 'Cluster 1', 'Cluster 2'])

plot_df_MX.plot(x='type', kind='barh', stacked=False, title='K-Means clustered data, with k=2, MX dataset')
plt.xlabel('Percentage in each cluster', fontsize=12)
plt.xlim([0, 100])
plt.ylabel('')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()

################################################################################
# select k=3 and use for visualization
kmeans = KMeans(3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(df_IL)
df_IL['kmeans_index']=np.array(kmeans.labels_) # add another column wih the cluster index
# df_IL.to_csv('IL_dataset_clean_with_cluster.csv')
index_zero_IL_K3 = (df_IL['kmeans_index'].values == 0).sum() # count number of rows in each cluster
index_one_IL_K3 = (df_IL['kmeans_index'].values == 1).sum()
index_two_IL_K3 = (df_IL['kmeans_index'].values == 2).sum()
# split the sorted df into 3 dataframes
df_IL_sorted = df_IL.sort_values('kmeans_index') # new dataset sorted by the cluster index
df_cluster_zero_IL = df_IL_sorted.iloc[0:index_zero_IL_K3 - 1, :]
df_cluster_one_IL = df_IL_sorted.iloc[index_zero_IL_K3:index_zero_IL_K3 + index_one_IL_K3 - 1, :]
df_cluster_two_IL = df_IL_sorted.iloc[index_zero_IL_K3 + index_one_IL_K3:, :]
# calc means of of each cluster
total_count_full_IL = df_IL.sum()
total_count_zero_IL = df_cluster_zero_IL.sum()
total_count_one_IL = df_cluster_one_IL.sum()
total_count_two_IL = df_cluster_two_IL.sum()

index_zero_partial_IL = total_count_zero_IL / total_count_full_IL
index_one_partial_IL = total_count_one_IL / total_count_full_IL
index_two_partial_IL = total_count_two_IL / total_count_full_IL
################################################################################
# create new df for bar plotting
plot_df_IL = pd.DataFrame([['Is Female', 100 * index_zero_partial_IL.Gender, 100 * index_one_partial_IL.Gender, 100 * index_two_partial_IL.Gender], ['Cough', 100 * index_zero_partial_IL.Cough, 100 * index_one_partial_IL.Cough, 100 * index_two_partial_IL.Cough],
                           ['Fever', 100 * index_zero_partial_IL.Fever, 100 * index_one_partial_IL.Fever, 100 * index_two_partial_IL.Fever], ['Sore throat', 100 * index_zero_partial_IL.Sore_throat, 100 * index_one_partial_IL.Sore_throat, 100 * index_two_partial_IL.Sore_throat],
                           ['Shortness of breath', 100 * index_zero_partial_IL.Shortness_of_breath, 100 * index_one_partial_IL.Shortness_of_breath, 100 * index_two_partial_IL.Shortness_of_breath], ['Head ache', 100 * index_zero_partial_IL.Head_ache, 100 * index_one_partial_IL.Head_ache, 100 * index_two_partial_IL.Head_ache],
                           ['Is 60 and over', 100 * index_zero_partial_IL.Age_60_and_over, 100 * index_one_partial_IL.Age_60_and_over, 100 * index_two_partial_IL.Age_60_and_over]],
                          columns=['type', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

plot_df_IL.plot(x='type', kind='barh', stacked=False, title='K-Means clustered data, with k=3, IL dataset')
plt.xlabel('Percentage in each cluster', fontsize=12)
plt.xlim([0, 100])
plt.ylabel('')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()

################################################################################
# MX dataset
kmeans.fit(df_MX)
df_MX['kmeans_index']=np.array(kmeans.labels_) # add another column wih the cluster index
# df_MX.to_csv('MX_dataset_clean_with_cluster.csv')
index_zero_MX_K3 = (df_MX['kmeans_index'].values == 0).sum() # count number of rows in each cluster
index_one_MX_K3 = (df_MX['kmeans_index'].values == 1).sum()
index_two_MX_K3 = (df_MX['kmeans_index'].values == 2).sum()
# split the sorted df into 3 dataframes
df_MX_sorted = df_MX.sort_values('kmeans_index') # new dataset sorted by the cluster index
df_cluster_zero_MX = df_MX_sorted.iloc[0:index_zero_MX_K3 - 1, :]
df_cluster_one_MX = df_MX_sorted.iloc[index_zero_MX_K3:index_zero_MX_K3 + index_one_MX_K3 - 1, :]
df_cluster_two_MX = df_MX_sorted.iloc[index_zero_MX_K3 + index_one_MX_K3:, :]
# calc means of of each cluster
total_count_full_MX = df_MX.sum()
total_count_zero_MX = df_cluster_zero_MX.sum()
total_count_one_MX = df_cluster_one_MX.sum()
total_count_two_MX = df_cluster_two_MX.sum()

index_zero_partial_MX = total_count_zero_MX / total_count_full_MX
index_one_partial_MX = total_count_one_MX / total_count_full_MX
index_two_partial_MX = total_count_two_MX / total_count_full_MX
################################################################################
# create new df for bar plotting
plot_df_MX = pd.DataFrame([['Is Female', 100 * index_zero_partial_MX.Gender, 100 * index_one_partial_MX.Gender, 100 * index_two_partial_MX.Gender], ['Cough', 100 * index_zero_partial_MX.Cough, 100 * index_one_partial_MX.Cough, 100 * index_two_partial_MX.Cough],
                           ['Fever', 100 * index_zero_partial_MX.Fever, 100 * index_one_partial_MX.Fever, 100 * index_two_partial_MX.Fever], ['Sore throat', 100 * index_zero_partial_MX.Sore_throat, 100 * index_one_partial_MX.Sore_throat, 100 * index_two_partial_MX.Sore_throat],
                           ['Shortness of breath', 100 * index_zero_partial_MX.Shortness_of_breath, 100 * index_one_partial_MX.Shortness_of_breath, 100 * index_two_partial_MX.Shortness_of_breath], ['Head ache', 100 * index_zero_partial_MX.Head_ache, 100 * index_one_partial_MX.Head_ache, 100 * index_two_partial_MX.Head_ache],
                           ['Is 60 and over', 100 * index_zero_partial_MX.Age_60_and_over, 100 * index_one_partial_MX.Age_60_and_over, 100 * index_two_partial_MX.Age_60_and_over]],
                          columns=['type', 'Cluster 1', 'Cluster 2', 'Cluster 3'])

plot_df_MX.plot(x='type', kind='barh', stacked=False, title='K-Means clustered data, with k=3, MX dataset')
plt.xlabel('Percentage in each cluster', fontsize=12)
plt.xlim([0, 100])
plt.ylabel('')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()

################################################################################
# select k=4 and use for visualization
kmeans = KMeans(4, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(df_IL)
df_IL['kmeans_index']=np.array(kmeans.labels_) # add another column wih the cluster index
# df_IL.to_csv('IL_dataset_clean_with_cluster.csv')
index_zero_IL_K4 = (df_IL['kmeans_index'].values == 0).sum() # count number of rows in each cluster
index_one_IL_K4 = (df_IL['kmeans_index'].values == 1).sum()
index_two_IL_K4 = (df_IL['kmeans_index'].values == 2).sum()
index_three_IL_K4 = (df_IL['kmeans_index'].values == 3).sum()
# split the sorted df into 4 dataframes
df_IL_sorted = df_IL.sort_values('kmeans_index') # new dataset sorted by the cluster index
df_cluster_zero_IL = df_IL_sorted.iloc[0:index_zero_IL_K4 - 1, :]
df_cluster_one_IL = df_IL_sorted.iloc[index_zero_IL_K4:index_zero_IL_K4 + index_one_IL_K4 - 1, :]
df_cluster_two_IL = df_IL_sorted.iloc[index_zero_IL_K4 + index_one_IL_K4: index_zero_IL_K4 + index_one_IL_K4 + index_two_IL_K4 - 1, :]
df_cluster_three_IL = df_IL_sorted.iloc[index_zero_IL_K4 + index_one_IL_K4 + index_two_IL_K4:, :]

# calc means of of each cluster
total_count_full_IL = df_IL.sum()
total_count_zero_IL = df_cluster_zero_IL.sum()
total_count_one_IL = df_cluster_one_IL.sum()
total_count_two_IL = df_cluster_two_IL.sum()
total_count_three_IL = df_cluster_three_IL.sum()

index_zero_partial_IL = total_count_zero_IL / total_count_full_IL
index_one_partial_IL = total_count_one_IL / total_count_full_IL
index_two_partial_IL = total_count_two_IL / total_count_full_IL
index_three_partial_IL = total_count_three_IL / total_count_full_IL
################################################################################
# create new df for bar plotting
plot_df_IL = pd.DataFrame([['Is Female', 100 * index_zero_partial_IL.Gender, 100 * index_one_partial_IL.Gender, 100 * index_two_partial_IL.Gender, 100 * index_three_partial_IL.Gender], ['Cough', 100 * index_zero_partial_IL.Cough, 100 * index_one_partial_IL.Cough, 100 * index_two_partial_IL.Cough, 100 * index_three_partial_IL.Cough],
                           ['Fever', 100 * index_zero_partial_IL.Fever, 100 * index_one_partial_IL.Fever, 100 * index_two_partial_IL.Fever, 100 * index_three_partial_IL.Fever], ['Sore throat', 100 * index_zero_partial_IL.Sore_throat, 100 * index_one_partial_IL.Sore_throat, 100 * index_two_partial_IL.Sore_throat, 100 * index_three_partial_IL.Sore_throat],
                           ['Shortness of breath', 100 * index_zero_partial_IL.Shortness_of_breath, 100 * index_one_partial_IL.Shortness_of_breath, 100 * index_two_partial_IL.Shortness_of_breath, 100 * index_three_partial_IL.Shortness_of_breath], ['Head ache', 100 * index_zero_partial_IL.Head_ache, 100 * index_one_partial_IL.Head_ache, 100 * index_two_partial_IL.Head_ache, 100 * index_three_partial_IL.Head_ache],
                           ['Is 60 and over', 100 * index_zero_partial_IL.Age_60_and_over, 100 * index_one_partial_IL.Age_60_and_over, 100 * index_two_partial_IL.Age_60_and_over, 100 * index_three_partial_IL.Age_60_and_over]],
                          columns=['type', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
# print(plot_df)
plot_df_IL.plot(x='type', kind='barh', stacked=False, title='K-Means clustered data, with k=4, IL dataset')
plt.xlabel('Percentage in each cluster', fontsize=12)
plt.xlim([0, 100])
plt.ylabel('')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()

################################################################################
kmeans.fit(df_MX)
df_MX['kmeans_index']=np.array(kmeans.labels_) # add another column wih the cluster index
# df_MX.to_csv('MX_dataset_clean_with_cluster.csv')
index_zero_MX_K4 = (df_MX['kmeans_index'].values == 0).sum() # count number of rows in each cluster
index_one_MX_K4 = (df_MX['kmeans_index'].values == 1).sum()
index_two_MX_K4 = (df_MX['kmeans_index'].values == 2).sum()
index_three_MX_K4 = (df_MX['kmeans_index'].values == 3).sum()
# split the sorted df into 4 dataframes
df_MX_sorted = df_MX.sort_values('kmeans_index') # new dataset sorted by the cluster index
df_cluster_zero_MX = df_MX_sorted.iloc[0:index_zero_MX_K4 - 1, :]
df_cluster_one_MX = df_MX_sorted.iloc[index_zero_MX_K4:index_zero_MX_K4 + index_one_MX_K4 - 1, :]
df_cluster_two_MX = df_MX_sorted.iloc[index_zero_MX_K4 + index_one_MX_K4: index_zero_MX_K4 + index_one_MX_K4 + index_two_MX_K4 - 1, :]
df_cluster_three_MX = df_MX_sorted.iloc[index_zero_MX_K4 + index_one_MX_K4 + index_two_MX_K4:, :]

# calc means of of each cluster
total_count_full_MX = df_MX.sum()
total_count_zero_MX = df_cluster_zero_MX.sum()
total_count_one_MX = df_cluster_one_MX.sum()
total_count_two_MX = df_cluster_two_MX.sum()
total_count_three_MX = df_cluster_three_MX.sum()

index_zero_partial_MX = total_count_zero_MX / total_count_full_MX
index_one_partial_MX = total_count_one_MX / total_count_full_MX
index_two_partial_MX = total_count_two_MX / total_count_full_MX
index_three_partial_MX = total_count_three_MX / total_count_full_MX
################################################################################
# create new df for bar plotting
plot_df_MX = pd.DataFrame([['Is Female', 100 * index_zero_partial_MX.Gender, 100 * index_one_partial_MX.Gender, 100 * index_two_partial_MX.Gender, 100 * index_three_partial_MX.Gender], ['Cough', 100 * index_zero_partial_MX.Cough, 100 * index_one_partial_MX.Cough, 100 * index_two_partial_MX.Cough, 100 * index_three_partial_MX.Cough],
                           ['Fever', 100 * index_zero_partial_MX.Fever, 100 * index_one_partial_MX.Fever, 100 * index_two_partial_MX.Fever, 100 * index_three_partial_MX.Fever], ['Sore throat', 100 * index_zero_partial_MX.Sore_throat, 100 * index_one_partial_MX.Sore_throat, 100 * index_two_partial_MX.Sore_throat, 100 * index_three_partial_MX.Sore_throat],
                           ['Shortness of breath', 100 * index_zero_partial_MX.Shortness_of_breath, 100 * index_one_partial_MX.Shortness_of_breath, 100 * index_two_partial_MX.Shortness_of_breath, 100 * index_three_partial_MX.Shortness_of_breath], ['Head ache', 100 * index_zero_partial_MX.Head_ache, 100 * index_one_partial_MX.Head_ache, 100 * index_two_partial_MX.Head_ache, 100 * index_three_partial_MX.Head_ache],
                           ['Is 60 and over', 100 * index_zero_partial_MX.Age_60_and_over, 100 * index_one_partial_MX.Age_60_and_over, 100 * index_two_partial_MX.Age_60_and_over, 100 * index_three_partial_MX.Age_60_and_over]],
                          columns=['type', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])

plot_df_MX.plot(x='type', kind='barh', stacked=False, title='K-Means clustered data, with k=4, MX dataset')
plt.xlabel('Percentage in each cluster', fontsize=12)
plt.xlim([0, 100])
plt.ylabel('')
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.tight_layout()
plt.show()








# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import jaccard_score
# from sklearn.cluster import KMeans
# ################################################################################
# # load Israel data
# filename_csv = 'Israel_dataset_clean.csv'
# df_IL = pd.read_csv(filename_csv)
# df_IL.drop(['Unnamed: 0'], axis=1, inplace=True) # drop this column
#
# # load mexico data
# filename_csv = 'clean_data_mx.csv'
# df_MX=pd.read_csv(filename_csv) #, delimiter=','
# df_MX.drop(['Unnamed: 0'], axis=1, inplace=True) # drop this column
#
# # replace in gender feature
# df_MX.replace('F', 1, inplace=True) # replace hebrew 'female' with 1
# df_MX.replace('M', 0, inplace=True) # replace hebrew 'male' with 0
#
# mean_of_columns_IL = df_IL.mean()
# mean_of_columns_MX = df_MX.mean()
# ################################################################################
#
# # filter out symptomatic samples IL
# df_asymptomatic_IL = df_IL[df_IL['Cough'] == 0]
# df_asymptomatic_IL = df_asymptomatic_IL[df_asymptomatic_IL['Fever'] == 0]
# df_asymptomatic_IL = df_asymptomatic_IL[df_asymptomatic_IL['Sore_throat'] == 0]
# df_asymptomatic_IL = df_asymptomatic_IL[df_asymptomatic_IL['Shortness_of_breath'] == 0]
# df_asymptomatic_IL = df_asymptomatic_IL[df_asymptomatic_IL['Head_ache'] == 0]
#
# df_asymptomatic_IL.drop(['Cough'], axis=1, inplace=True) # drop this column
# df_asymptomatic_IL.drop(['Fever'], axis=1, inplace=True) # drop this column
# df_asymptomatic_IL.drop(['Sore_throat'], axis=1, inplace=True) # drop this column
# df_asymptomatic_IL.drop(['Shortness_of_breath'], axis=1, inplace=True) # drop this column
# df_asymptomatic_IL.drop(['Head_ache'], axis=1, inplace=True) # drop this column
#
# # filter out symptomatic samples MX
# df_asymptomatic_MX = df_MX[df_MX['Cough'] == 0]
# df_asymptomatic_MX = df_asymptomatic_MX[df_asymptomatic_MX['Fever'] == 0]
# df_asymptomatic_MX = df_asymptomatic_MX[df_asymptomatic_MX['Sore_throat'] == 0]
# df_asymptomatic_MX = df_asymptomatic_MX[df_asymptomatic_MX['Shortness_of_breath'] == 0]
# df_asymptomatic_MX = df_asymptomatic_MX[df_asymptomatic_MX['Head_ache'] == 0]
#
# df_asymptomatic_MX.drop(['Cough'], axis=1, inplace=True) # drop this column
# df_asymptomatic_MX.drop(['Fever'], axis=1, inplace=True) # drop this column
# df_asymptomatic_MX.drop(['Sore_throat'], axis=1, inplace=True) # drop this column
# df_asymptomatic_MX.drop(['Shortness_of_breath'], axis=1, inplace=True) # drop this column
# df_asymptomatic_MX.drop(['Head_ache'], axis=1, inplace=True) # drop this column
#
# mean_of_columns_asymptomatic_IL = df_asymptomatic_IL.mean()
# mean_of_columns_asymptomatic_MX = df_asymptomatic_MX.mean()
#
#
# ## Symptomatic
# df_Symptomatic_IL = df_IL[df_IL['Cough'] == 1]
# df_Symptomatic_IL = df_Symptomatic_IL[df_Symptomatic_IL['Fever'] == 1]
# df_Symptomatic_IL = df_Symptomatic_IL[df_Symptomatic_IL['Sore_throat'] == 1]
# df_Symptomatic_IL = df_Symptomatic_IL[df_Symptomatic_IL['Shortness_of_breath'] == 1]
# df_Symptomatic_IL = df_Symptomatic_IL[df_Symptomatic_IL['Head_ache'] == 1]
#
# df_Symptomatic_IL.drop(['Cough'], axis=1, inplace=True) # drop this column
# df_Symptomatic_IL.drop(['Fever'], axis=1, inplace=True) # drop this column
# df_Symptomatic_IL.drop(['Sore_throat'], axis=1, inplace=True) # drop this column
# df_Symptomatic_IL.drop(['Shortness_of_breath'], axis=1, inplace=True) # drop this column
# df_Symptomatic_IL.drop(['Head_ache'], axis=1, inplace=True) # drop this column
#
# # filter out symptomatic samples MX
# df_Symptomatic_MX = df_MX[df_MX['Cough'] == 1]
# df_Symptomatic_MX = df_Symptomatic_MX[df_Symptomatic_MX['Fever'] == 1]
# df_Symptomatic_MX = df_Symptomatic_MX[df_Symptomatic_MX['Sore_throat'] == 1]
# df_Symptomatic_MX = df_Symptomatic_MX[df_Symptomatic_MX['Shortness_of_breath'] == 1]
# df_Symptomatic_MX = df_Symptomatic_MX[df_Symptomatic_MX['Head_ache'] == 1]
#
# df_Symptomatic_MX.drop(['Cough'], axis=1, inplace=True) # drop this column
# df_Symptomatic_MX.drop(['Fever'], axis=1, inplace=True) # drop this column
# df_Symptomatic_MX.drop(['Sore_throat'], axis=1, inplace=True) # drop this column
# df_Symptomatic_MX.drop(['Shortness_of_breath'], axis=1, inplace=True) # drop this column
# df_Symptomatic_MX.drop(['Head_ache'], axis=1, inplace=True) # drop this column
#
# mean_of_columns_Symptomatic_IL = df_Symptomatic_IL.mean()
# mean_of_columns_Symptomatic_MX = df_Symptomatic_MX.mean()
#
#
