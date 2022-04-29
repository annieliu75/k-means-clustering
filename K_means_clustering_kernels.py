#K-CLUSTERING FUNCTION WITH KERNEL
import numpy as np
from kernels_functions import kernel


def K_means_clustering_kernel(X,K,kernel_name):
  """
  K-means clustering with kernelisation, we use the kernel trick
  input : X : data
          K : numbers of clusters
          kernel_name : name of the kernel we want to use
  output : clusters 
  """
  print("K means clustering on your data using " + kernel_name)

  #Initialize clusters , we do not compute here the centroids directly
  N=len(X)
  indexes=[i for i in range(N)]
  clusters=[indexes[i::K] for i in range(K)] #we create the clusters by making a K-split on all indexes

  new_clusters=asign_new_clusters(clusters,X,K,kernel_name)

  compteur=1 #count the number of iterations
  #Normally we would use the loss function, but we cannot compute the loss directly since we do not have phi, nor we compute the centroids directly, but in fact since it is always getting better at each iteration, we just have to loop until clusters aren't changing anymore => no more improvement
  while not convergence_clusters(new_clusters,clusters,K): #see if two clusters are equals
    ## uncomment this if we are testing on simpler data set such a circles, etc,... and we want to visualize the progression
    #  print_clusters(K,clusters,X)
    compteur +=1
    clusters=new_clusters.copy() 
    new_clusters=asign_new_clusters(clusters,X,K,kernel_name) 

  print("Iteration number : " + str(compteur))
  return clusters

def asign_new_clusters(clusters, X,K,kernel_name): 
  """
  we asign each x_j to cluster 
  k = argmin_l {  K(xj,xj) - 2*sum(K(xj,xi))/card + sum(K(xi,xk))/card**2  } 
  #we use here the kernel trick so we don't have to compute centroids
  """
  new_clusters=[[] for i in range(K)] #placeholder
  for i in range(len(X)): #for each xi
    new_clusters[argmin_l(X[i],X,clusters,K,kernel_name)].append(i)
  return new_clusters

def argmin_l(xj,X,clusters,K,kernel_name):
  """
  Assign each xj to cluster k = arg min (pseudo_loss)
  where  pseudo_loss= K(xj,xj) - 2*sum(K(xj,xi))/card + sum(K(xi,xk))/card**2
  """
  delta=10e-4 #since we divide by the len of clusters, it is to make sure we won't have any problems with empty clusters
  pseudo_loss=[] #placeholder
  for l in range(K):
    dist_from_ker=sum([kernel(xj,X[i],kernel_name) for i in clusters[l]])
    dist_in_ker=sum([kernel(X[i],X[k],kernel_name) for i in clusters[l] for k in clusters[l]])
    pseudo_loss.append(  
        kernel(xj,xj,kernel_name) - 2*dist_from_ker/(len(clusters[l])+delta) + dist_in_ker/((len(clusters[l])+delta)**2) # fonction to optimize to get the new cluster attribuition
    ) 
  return np.argmin(pseudo_loss)

def convergence_clusters(new_clusters,clusters,K): 
  """
  Compares two clusters and tells if they are the same
  """
  flag=True 
  for l in range(K):
    if new_clusters[l]!=clusters[l]: #if at least one clusters is still changing (so in fact 2)
      flag=False
  return flag