import numpy as np

#KERNEL functions 


#Here we propose using some existing kernels, but there are many kernels, for example we could also try with these kernels
#Others kernels ideas such as 
#Fischl
#Rational Quadratic Kernel
#ANOVA Kernel
#Multiquadric Kernel
#Inverse Multiquadric Kernel
#Circular Kernel
#Spherical Kernel
#Wave Kernel
#Power Kernel
#Log Kernel
#Spline Kernel
#B-Spline (Radial Basis Function) Kernel
#Bessel Kernel
#Cauchy Kernel
#Chi-Square Kernel
#Histogram Intersection Kernel
#Generalized Histogram Intersection
#Generalized T-Student Kernel
#Bayesian Kernel
#Wavelet Kernel
#cf : http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#laplacian 


def kernel(x,y,kernel_name,Sigma=1.,K_sig=1.,delta_sig=0.,c_pol=0.,h_pol=2.): 
  """
  Compute and return the corresponding kernel giving the name
  entries : vectors x and y , kernel_name and kernel parameters
  output : scalar kernel

  available kernel names : 
  kernel_name='Polynomial Kernel' # 'Gaussian Kernel', 'Sigmoid Kernel' ; 'Polynomial Kernel', 'No Kernel'

  note : Sigma is for Gaussian and Laplace kernel
         K_sig and delta_sig for sigmoid kernel
         c_pol and h_pol for polynomial kernel
  """
  if kernel_name=='Gaussian Kernel':
    return Gaussian_Kernel(x,y,Sigma)
  elif kernel_name=='Sigmoid Kernel':
    return Sigmoid_Kernel(x,y,K_sig,delta_sig)
  elif kernel_name=='Polynomial Kernel':
    return Polynomial_Kernel(x,y,c_pol,h_pol)
  elif kernel_name=='Laplace Kernel':
    return Laplace_Kernel(x,y,Sigma)
  elif kernel_name=='No Kernel':
    return no_kernel(x,y)
  else :
    print("Error, we didn't recognise the kernel name")
  
def no_kernel(x,y): #euclidean, should be the same without kernels
  return np.dot(x,y)

def Gaussian_Kernel(x,y,Sigma):
  return np.exp(-np.inner(x-y,x-y)/(2*Sigma**2)) 
  
def Sigmoid_Kernel(x,y,K_sig,delta_sig):
  return np.tanh(K_sig*(np.dot(x,y))-delta_sig)

def Polynomial_Kernel(x,y,c_pol,h_pol):
  return (np.dot(x,y)+c_pol)**h_pol

def Laplace_Kernel(x,y,Sigma):
  return np.exp(-np.linalg.norm(x-y)/(Sigma)) 
