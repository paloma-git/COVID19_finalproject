import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.manifold import TSNE  # T-Distributed Stochastic Neighbor Embedding
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Read csv file and import as dataframe
# filename_csv = 'clean_data_mx.csv'
filename_csv = 'Israel_dataset_clean.csv'
# filename_csv = 'MX_ISR_clean.csv'
df = pd.read_csv(filename_csv)

# print(df.head())

if filename_csv == 'clean_data_mx.csv':
    df.drop(columns='Unnamed: 0', axis=1, inplace=True)
    df = df.rename(
        columns={'tos': 'Cough', 'fiebre': 'Fever', 'odinogia': 'Sore Throat', 'disnea': 'Shortness of Breath',
                 'cefalea': 'Headache', 'edad': 'Age', 'sexo': 'Gender'})
    df['Gender'].replace({'F': 1, 'M': 0}, inplace=True)
    # Find ONLY symptomatic samples
    df = df.loc[((df['Cough'] == 1) | (df['Fever'] == 1) | (df['Sore Throat'] == 1) | (
            df['Shortness of Breath'] == 1) | (df['Headache'] == 1)), :]
    print(df.shape)

elif filename_csv == 'MX_ISR_clean.csv':
    df.drop(columns='Unnamed: 0', axis=1, inplace=True)
    df.drop(columns='country_index', axis=1, inplace=True)
    # df.drop(columns='Gender', axis=1, inplace=True)

elif filename_csv == 'Israel_dataset_clean.csv':
    df.drop(columns='Unnamed: 0', axis=1, inplace=True)
    df.drop(columns='country_index', axis=1, inplace=True)
    # Find ONLY symptomatic samples
    df = df.loc[((df['cough'] == 1) | (df['fever'] == 1) | (df['sore_throat'] == 1) | (
                df['shortness_of_breath'] == 1) | (df['head_ache'] == 1)), :]
    df.astype(float)
    print(df.shape)

mat = df.values

km = sklearn.cluster.KMeans(n_clusters=2)
km.fit(mat)

# Find which cluster each data-point belongs to
clusters = km.predict(mat)

# PCA with one principal component
pca_1d = PCA(n_components=1)

# PCA with two principal components
pca_2d = PCA(n_components=2)

# PCA with three principal components
pca_3d = PCA(n_components=3)

X = df
X["Cluster"] = clusters

# plotX is a DataFrame containing 5000 values sampled randomly from X
plotX = pd.DataFrame(np.array(X.sample(10000)))

# Rename plotX's columns since it was briefly converted to an np.array above
plotX.columns = X.columns

# This DataFrame holds that single principal component mentioned above
PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

# This DataFrame contains the two principal components that will be used
# for the 2-D visualization mentioned above
PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

# And this DataFrame contains three principal components that will aid us
# in visualizing our clusters in 3-D
PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

PCs_1d.columns = ["PC1_1d"]

# "PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
# And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
PCs_2d.columns = ["PC1_2d", "PC2_2d"]

PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

# We concatenate these newly created DataFrames to plotX so that they can be used by plotX as columns.
plotX = pd.concat([plotX, PCs_1d, PCs_2d, PCs_3d], axis=1, join='inner')

# Note that all of the DataFrames below are sub-DataFrames of 'plotX'.
# This is because we intend to plot the values contained within each of these DataFrames.

cluster0 = plotX[plotX["Cluster"] == 0]
cluster1 = plotX[plotX["Cluster"] == 1]
cluster2 = plotX[plotX["Cluster"] == 2]

# Instructions for building the 3-D plot

# trace1 is for 'Cluster 0'
trace1 = go.Scatter3d(
    x=cluster0["PC1_3d"],
    y=cluster0["PC2_3d"],
    z=cluster0["PC3_3d"],
    mode="markers",
    name="Cluster 0",
    marker=dict(color='rgba(255, 128, 255, 0.8)'),  # pink
    text=None)

# trace2 is for 'Cluster 1'
trace2 = go.Scatter3d(
    x=cluster1["PC1_3d"],
    y=cluster1["PC2_3d"],
    z=cluster1["PC3_3d"],
    mode="markers",
    name="Cluster 1",
    marker=dict(color='rgba(255, 128, 2, 0.8)'),  # orange
    text=None)

# trace3 is for 'Cluster 2'
trace3 = go.Scatter3d(
    x=cluster2["PC1_3d"],
    y=cluster2["PC2_3d"],
    z=cluster2["PC3_3d"],
    mode="markers",
    name="Cluster 2",
    marker=dict(color='rgba(0, 150, 200, 0.8)'),  # blue
    text=None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Three Dimensions Using PCA"

layout = dict(title=title,
              xaxis=dict(title='PC1', ticklen=5, zeroline=False),
              yaxis=dict(title='PC2', ticklen=5, zeroline=False)
              )

fig = dict(data=data, layout=layout)

iplot(fig)

# Instructions for building the 2-D plot

# trace1 is for 'Cluster 0'
trace1 = go.Scatter(
    x=cluster0["PC1_2d"],
    y=cluster0["PC2_2d"],
    mode="markers",
    name="Cluster 0",
    marker=dict(color='rgba(255, 128, 255, 0.8)', symbol='circle', size=15),
    text=None)

# trace2 is for 'Cluster 1'
trace2 = go.Scatter(
    x=cluster1["PC1_2d"],
    y=cluster1["PC2_2d"],
    mode="markers",
    name="Cluster 1",
    marker=dict(color='rgba(255, 128, 2, 0.8)', symbol='square', size=15),
    text=None)

# trace3 is for 'Cluster 2'
trace3 = go.Scatter(
    x=cluster2["PC1_2d"],
    y=cluster2["PC2_2d"],
    mode="markers",
    name="Cluster 2",
    marker=dict(color='rgba(0, 150, 200, 0.8)', symbol='diamond', size=15),
    text=None)

data = [trace1, trace2, trace3]

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title=title,
              xaxis=dict(title='PC1', ticklen=5, zeroline=False),
              yaxis=dict(title='PC2', ticklen=5, zeroline=False)
              )

fig = dict(data=data, layout=layout)

iplot(fig)

c0 = X[X["Cluster"] == 0]
c1 = X[X["Cluster"] == 1]
c2 = X[X["Cluster"] == 2]

print('Cluster 0 statistics: ')
print('Samples in this cluster: ', c0.shape)
print(c0.sum(axis=0))

print('Cluster 1 statistics: ')
print('Samples in this cluster: ', c1.shape)
print(c1.sum(axis=0))

print('Cluster 2 statistics: ')
print('Samples in this cluster: ', c2.shape)
print(c2.sum(axis=0))
