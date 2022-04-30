#For normalized and unormalized spectral clustering on MNIST dataset

import numpy as np
import matplotlib.pyplot as plt

def similarity_matrix(X):
  """
  Computes the similarity matrix of X, we here use gaussian kernel, This is already tuned for MNIST dataset
  """
  N=len(X)
  W = np.zeros([N,N])
  gamma=9e-10 # determined with 700 samples
  for i in range(N):
    for j in [x for x in range(N) if x != i]:
      temp = np.inner( (X[i]-X[j]), (X[i]-X[j]))*gamma
      W[i,j] = np.exp(-temp/2)
  return W

def laplacian_matrix(X):
  """
  Computes the normalised laplacian matrix of X
  """
  N=len(X)
  I=np.identity(N) #comment for unormalized
  W=similarity_matrix(X)
  D = np.zeros([N,N])
  for i in range(N):
    for j in range(N):
      D[i,i] = D[i,i]+W[i,j]
      D[i,i]=D[i,i]**(-0.5) #comment for unormalized
  #L = D - W #uncomment this line for unnormalized and ucomment the next line 
  L = I - D @ W @ D #comment for un normalized 
  return L

def H_constrcution(X,K,print_first_eig_val=False):
  """
  Computes H whose columns are the eigenvectors corresponding to the K minimal eigenvalues of L.
  """
  N=len(X)
  H = np.zeros([K,N])
  L = laplacian_matrix(X)
  EV,V = np.linalg.eig(L) #we get the EV : eigen values, and V : eign vectors of Laplacian
  V = np.real(V) #to have real only
  EV=np.abs(EV) #we are interested in how close they are to 0
  idx = EV.argsort()  
  EV=EV[idx]
  V=V[:,idx]
  if print_first_eig_val: #if we want to print first eigen values
    plt.plot(range(100),EV[0:100], 'rx')
  H = V[:, :K] #first K vects
  return H