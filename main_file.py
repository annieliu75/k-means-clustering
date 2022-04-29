#main file for the project

###imports
import sklearn 
from sklearn.datasets import fetch_openml #for MNIST digits data set
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs

import print_img
from K_means_clustering_SED import K_means_clustering_standard_SED
from K_means_clustering_SED import asign_clusters
from K_means_clustering_kernels import K_means_clustering_kernel
from accuracy import accuracy
### Parameters & settings

k_mean_mode= "SED" #"SED" ,"Kernelized"
data_set= 'digits' # 'digits', 'circle' , 'moon', 'blobs' #select the type of the data set you want to test

N_sample=300 #Size of the data, if on all data change to "all_data" : but only if we are working on MNIST 
K=10 #number of clusters

#Parameters for the kernelized mode : 
kernel_name='Gaussian Kernel' # 'Gaussian Kernel', 'Sigmoid Kernel' ; 'Polynomial Kernel', 'No Kernel', 'Laplace Kernel'

#Kernels parameters are set by default, if you want to change them see in kernels_functions.py file


### Main function 
if __name__ == '__main__':
    #take the dataset and labels
    if N_sample=='all data': #if all_data, we automatically set to the data_set digits from mnist
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X=X.values #convert pandas, dataframe to numpy array
        y=y.values
    else :
        if data_set=='digits':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            X=X.values[0:N_sample]
            y=y.values[0:N_sample]  
        elif data_set==  'circle':
            X, y = make_circles(n_samples=N_sample, factor=0.01, noise=0.05, random_state=0)
            K=2 #to avoid too many clusters when testing on circles, since it is 2 clusters
        elif data_set==  'blobs':
            X, y= make_blobs(n_samples=N_sample,centers=K,cluster_std=0.6, random_state=0)
        elif data_set=='moon':
            X, y = make_moons(n_samples=N_sample, noise=0.035, random_state=0)
            K=2 #to avoid too many clusters when testing on moons, since it is 2 clusters
        else :
            print("Error, we did not recognize data_set type")
            

    if k_mean_mode=="SED":

        centroids=K_means_clustering_standard_SED(X,K)
        clusters = asign_clusters(centroids,X,K)
        if data_set== "digits" :
            print("Here are what the centroids looks like : ")
            print_img.print_digit(centroids)
            print("Here are what a sample of the clusters looks like : ")
            print_img.print_digit_clusters(X,clusters,K)    
        else :
            print_img.print_2D_clusters(X,clusters,K)    
        print("Using the k-mean clustering without kernel on our dataset we were able to get an accuracy of " + str(accuracy(clusters,y,K)))
    
    elif k_mean_mode=="Kernelized":
        """
        Remark: If we want to change the kernel parameters,
        change directly in the kernels_functions_.py file
        """
        clusters=K_means_clustering_kernel(X,K,kernel_name)
        
        if data_set== "digits" :
            print("Here are what a sample of the clusters looks like : ")
            print_img.print_digit_clusters(X,clusters,K)    
        else :
            print_img.print_2D_clusters(X,clusters,K)    
        print("Using the k-mean clustering with "+ kernel_name+ " on our dataset we were able to get an accuracy of " + str(accuracy(clusters,y,K)))