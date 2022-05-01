def QR_algo(A):
  """
  Function to compute eigenvectors and eigenvalues of A
  input : A matrix n x n 
  output : S and D : eigenvectors in S  and values in the diag of D
  """
  #Parameters : TOL (tolerance)
  TOL=10e-2
  N=1000 #another criteria.

  #Initialise variables and matrices
  k=1
  D = np.diag(np.diag(A))
  L = np.tril(A)-D
  Q,R = np.linalg.qr(A)

  S = np.identity(len(Q))

  while np.linalg.norm(L)>TOL and k<=N: #Stop when L is lower triangular
    Q,R=QR_factorisation(A) #QR factorize A
    S = np.dot(S,Q) #S is product of all previous Q

    A = np.dot(R,Q) #Define new A iteration
        
    D = np.diag(np.diag(A)) #D will contain eigenvalues
    L = np.tril(A)-D
    U = np.triu(A)-D
    k+=1

    
  return S,D

def QR_factorisation(A):
  """
  Function to compute QR factorsation of matrix A
  input: matrix A
  output: QR factorisation of A
  """
  n,m = A.shape

  Q = np.empty((n,n))
  X = np.empty((n,n)) #Auxilliary array

  Q[:,0] = A[:,0]/np.linalg.norm(A[:,0])
  X[:,0] = A[:,0]

  #Iteration of Gram-Schmidt procedure
  for i in range(1,n):
    X[:,i] = A[:,i]
    for j in range(i):
      X[:,i] = X[:,i] - np.inner(A[:,i],Q[:,j])*Q[:,j] #Orthogonalization

    Q[:,i] = X[:,i]/np.linalg.norm(X[:,i]) #Normalize
  
  #Define R matrix using found Q
  R = np.zeros((n,m))
  for i in range(n):
    for j in range(m):
      R[i,j] = np.inner(A[:,j],Q[:,i])

  return Q,R

def QR_Eigensolver(A):
  """
  Function that uses QR algorithm to find eigenvalues/vectors of A
  input: Matrix A
  output: eigenvalues, eigenvectors as columns
  """

  Q,D = QR_algo(A)
  ordering = np.argsort(abs(np.diag(D))) #Sort eigenvalues from small to large
  #Return sorted eigenvalues and sorted column vectors
  return np.diag(D)[ordering], Q[:,ordering]
