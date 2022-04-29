#others functions used for others things ...
import matplotlib.pyplot as plt
from IPython.display import clear_output #for clear output to have dynamic images


# If you want to use this, just add this additionnal line in K_means_clustering_kernels line 27 and import this file
# with from utils import print_clusters 
# Please, only use this when testing on 2D data!
def print_clusters(K,clusters,X):
  """
  print visually clusters and clear the old one : usefull if we want to visualise how the clusters are changing when performing on sklearn data set
  only for 2D points
  """
  clear_output()
  cols=['k','r','y','g','b','c','m','k','w'] #number of color limited, add new colors for more clusters
  plt.figure()
  for l in range(K):
      for point in clusters[l]:
          plt.scatter(X[point, 0],X[point, 1],  c=cols[l])
  plt.show()


