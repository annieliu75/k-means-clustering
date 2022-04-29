import matplotlib.pyplot as plt

def print_digit_clusters(X,clusters,K):
    """
    visualize a sample of all clusters
    """
    for L in range(K):
        plt.figure(L)
        
        fig, axs = plt.subplots(5, 5, figsize=(5, 5))
        X_l=X[clusters[L]]
        for i in range(min(len(X_l),25)): #so if a clusters is too small, we do not enconter a problem trying to go beyond what is in the cluster
            ax = axs[i // 5][i % 5]
            ax.imshow(X_l[i].reshape(28,28), cmap='gray') 
            ax.axis('off')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def print_digit(X):
    """
    visualize a sample of the digits // or all if less than 25
    """
    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    for i in range(min(len(X),25)):
        ax = axs[i // 5][i % 5]
        ax.imshow(X[i].reshape(28,28), cmap='gray')
        ax.axis('off')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def print_2D_clusters(X,clusters,K):
    cols=['k','r','y','g']
    plt.figure()
    for l in range(K):
        for point in clusters[l]:
            plt.scatter(X[point, 0],X[point, 1],  c=cols[l])
    plt.show()