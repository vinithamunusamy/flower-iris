#dimensional problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn .datasets import load_iris
from sklearn.preprocessing import StandardScaler
# load iris datasets
iris = load_iris()
x = iris.data
y = iris.target
target_names = iris.target_names
# scale the features (important for PCA)
x_scaled= StandardScaler().fit_transform(x)
# Apply PCA (reduce 4D->2D)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print("Explained varience Ratio:",pca.explained_variance_ratio_)
#plot the reduced 2D data
plt.figure(figsize=(6,5))
for i, target_name in enumerate(target_names):
  plt.scatter(x_pca[y==i,0],x_pca[y==i,1],label=target_name)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset(4D ->2D)")
plt.legend()
plt.show()
