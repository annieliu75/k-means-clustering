def Power_method(A,v,m, k):
  """
  Performs the power method on matix A, using vector v, for m iterations
  obtaining the k largest eigenvalues.
  input: matrix A, vector v, max. matrix power m, number of desired eigenvectors k
  output: matrix Q with eigenvectors, eigenvalues
  """
  Q = np.zeros((len(v), k)) 
  eigv = np.ones((k))
  v = v/np.linalg.norm(v) #normalise random vector
  v_tmp = v #used for iterations
  A_tmp = A #used for iterations
  for i in range(k):
    for it in range(m): #iterating matrix powers
      v_tmp = A_tmp@v_tmp
    Q[:,i] = v_tmp/np.linalg.norm(v_tmp) #normalise eigenvector
    eigv[i] = np.dot(A@Q[:,i],Q[:,i]) #find eigenvalue
    v_tmp = v
    for j in range(i):
      #remove already found eigenspace
      #this part does not function properly
      A_tmp = A_tmp - np.dot(A@Q[:,j],Q[:,j])*np.outer(Q[:,j],Q[:,j])
      v_tmp = v_tmp - np.dot(Q[:,j],v_tmp)
      v_tmp = v_tmp/np.linalg.norm(v_tmp)



  return Q, eigv


def Power_Eigensolver(A, q):
  """
  Eigensolver to find eigenvectors of q smallest eigenvalues.
  input: sparse symmetric matrix A, number of desired eigenvalues q<<N
  output: eigenvalues and eigenvalues

  """

  epsilon = 0.01 #used to regularize A
  m = 5 #number of iterations of for power method 

  v = (np.random.rand((len(A)))) #random vector for power solver

  Inv = np.linalg.inv(A+epsilon*np.identity(len(A)))
  #We determine 10q eigenvalues for accuracy
  Q, eigv = Power_method(Inv, v, m, k=10*q) 

  #Q contains basis of subspace corresponding to smallest eigenvalues of A
  
  eigenVectors = np.array([Q[:,vec] for vec in range(10*q)])

  eigenValues = eigv #np.zeros(len(eigenVectors))

  #Sorting eigenvectors; largest eignvl of inverse, are smallest eigenvl of A
    
  eigenVectors=eigenVectors[[np.flip(np.argsort(np.absolute(eigenValues)))]]
  for idx in range(len(eigenVectors)):   
    Q[:,idx] = eigenVectors[idx]
  eigenValues = eigenValues[np.flip(np.argsort(np.absolute(eigenValues)))]

  #taking q smallest eigenvalues and discarding degenerate ones
  eigenValues = np.unique(np.around(eigenValues[0:q], 5))
  Q = np.unique(np.around(Q[:,0:q], 5), axis=1)

  return 1/eigenValues, Q