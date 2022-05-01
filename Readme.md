## K-means clustering

This is the project for the course "Linear algebra and optimization for Machine Learning"
We here attempt to implement K-mean clustering :
 - with standard squared Euclidean distance
 - kernelized distance with some kernels (and using the "kernel" trick)  

  
 - unormalized and normalized spectral clustering
 - implement our eigensolver (for K-smallest eigenvalues, and in certains conditions on L laplacian matrix)

## How to use the code
1- set the setting in main_file.py, here you can set the mode, the data set, the number of sample we want to perform on (change to "all_data" if you want to run it on all data), the kernel you want to use

2- note : if you want to change kernel parameters go in kernels_functions.py and change it manually
[       Sigma is for Gaussian and Laplace kernel
         K_sig and delta_sig for sigmoid kernel
         c_pol and h_pol for polynomial kernel
         and by default they are set to 
         Sigma=1.,K_sig=1.,delta_sig=0.,c_pol=0.,h_pol=2.  ]

3- to run the code do :
```shell script
    python3 main_file.py
```
or use main.ipynb

## Alternative with notebooks
You can also find in the file notebooks, for the corresponding questions of the project and some tests on 2D data for example....


## Results
Here are some examples of results you should get with these parameters :
1- SED 
```
Parameters
```
```
Results
```
2- Kernelized
```
Parameters
```
```
Results
```
3- Spectral Clustering
```
Parameters
```
```
Results
```
#Number of clusters 
By default when clustering on MNIST data set we would always put the number of clusters to K=10, but we got better results if we put it to K=20 for example...  
It can be explained by the fact a same digit can be writed differently [example 0 or O], or writed in diagonal and so on...


## Version
This version of the code should work on python 3.10.4  
On older version, one of the issue you can enconter is that when importing MNIST data set, images and labels are already numpy array type, in that case change in main_file.py :
- comment line 37 and 38
- in line 42 and 43, change to 
```X=X.values[0:N_sample]
y=y.values[0:N_sample]  
```

## There may still be some issues when trying with differents parameters, we haven't been able to try every combinaison yet
