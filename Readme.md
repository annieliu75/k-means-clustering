## K-means clustering

This is the project for the course "Linear algebra and optimization for Machine Learning"
We here attempt to code K-mean clustering :
 - with standard squared Euclidean distance
 - kernelized distance with some kernels (and using the "kernel" trick)

 - unormalized and normalized spectral clustering

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
    python main_file.py
```
or use main.ipynb

## Alternative with notebooks
You can also find in the file notebooks, for the corresponding questions of our code.