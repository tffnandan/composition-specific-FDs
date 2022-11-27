from numpy import unique
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import scipy
import warnings
import os, sys

#data file
Data= pd.read_excel('int.xlsx')
ds = pd.DataFrame(Data)

#composition-specific clustering
#composition dataframe of k_tw/k_blt: k_car/k_blt:k_auto/k_blt
df1=ds.iloc[:,1:4]

#optimum clusters identifications using compositions

plt.figure(figsize=(8, 6))
n_components = np.arange(1, 15)
clfs = [GaussianMixture(n, max_iter = 1000).fit(df1) for n in n_components]
bics = [clf.bic(df1) for clf in clfs]
aics = [clf.aic(df1) for clf in clfs]

plt.plot(n_components, bics, label = 'BIC')
plt.plot(n_components, aics, label = 'AIC')
plt.xlabel('n_components')
plt.legend()
plt.show()

#Classification using GMM model
model = GaussianMixture(n_components=2, n_init=100, random_state= 100 )
model.fit(df1)
yhat = model.predict(df1)
clusters = unique(yhat)

#adding classes of the composition to a new column of the original dataframe
df2 = pd.DataFrame(yhat, columns = ['class1'],dtype = float)
ds.insert(7, "class1", df2['class1'], True)



#traffic-state-specific clustering
#traffic-state dataframe of flow-density-speed
df3=ds.iloc[:,6:9]

#optimum clusters identifications using traffic state

plt.figure(figsize=(8, 6))
n_components_1 = np.arange(1, 15)
clfs = [GaussianMixture(n, max_iter = 1000).fit(df3) for n in n_components_1]
bics = [clf.bic(df3) for clf in clfs]
aics = [clf.aic(df3) for clf in clfs]

plt.plot(n_components_1, bics, label = 'BIC')
plt.plot(n_components_1, aics, label = 'AIC')
plt.xlabel('n_components_1')
plt.legend()
plt.show()

#Classification using GMM model
model = GaussianMixture(n_components=3, n_init=100, random_state= 100 )
model.fit(df3)
yhat_1 = model.predict(df3)
clusters = unique(yhat_1)

#adding classes of the traffic-state to a new column of the original dataframe
df4 = pd.DataFrame(yhat, columns = ['class2'],dtype = float)
ds.insert(8, "class2", df4['class2'], True)

#combining "class1" and "class2"
# Dataframe of total-occupancy and flow
df=ds[['Q','total_occu']]

df1=df[(ds['class1']==0)|(ds['class2']==0)]
df2=df[(ds['class1']==0)|(ds['class2']==1)]
df3=df[(ds['class1']==0)|(ds['class2']==2)]
df4=df[(ds['class1']==1)|(ds['class2']==0)]
df5=df[(ds['class1']==1)|(ds['class2']==1)]
df6=df[(ds['class1']==1)|(ds['class2']==2)]

#plot k-mean points of all the combined class and join unique composition all k-mean points for different composition-specific FD
plt.figure(figsize=(12, 8))
#class A1
X1=df1
kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X1)
plt.subplot(1, 1, 1),plt.scatter(X1['total_occu'], X1['Q'],s=20,marker='D', color='orange',label='A1B2')
plt.subplot(1, 1, 1),plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, marker='D', color='orange',label='mean-A1B2')

X2=df2
kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X2)
plt.subplot(1, 1, 1),plt.scatter(X2['total_occu'], X2['Q'],s=20, marker='D', color='red',label='A1B3')
plt.subplot(1, 1, 1),plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, marker='D', color='red',label='mean-A1B3')

X3=df3
kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X3)
plt.subplot(1, 1, 1),plt.scatter(X3['total_occu'], X3['Q'],s=20,marker='D', color='grey',label='A1B3')
plt.subplot(1, 1, 1),plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, marker='D', color='grey',label='mean-A1B3')

#class A2
X4=df4
kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X4)
plt.subplot(1, 1, 1),plt.scatter(X4['total_occu'], X4['Q'],s=20,marker='^', color='black',label='A2B2')
plt.subplot(1, 1, 1),plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, marker='^', color='black',label='mean-A2B2')

X5=df5
kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X5)
plt.subplot(1, 1, 1),plt.scatter(X5['total_occu'], X5['Q'],s=20,marker='^', color='green',label='A2B1')
plt.subplot(1, 1, 1),plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, marker='^', color='green',label='mean-A2B1')


X6=df6
kmeans = KMeans(n_clusters=1, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X6)
plt.subplot(1, 1, 1),plt.scatter(X6['total_occu'], X6['Q'],s=20,marker='^', color='blue',label='A2B3')
plt.subplot(1, 1, 1),plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=100, marker='^', color='blue',label='mean-A2B3')







