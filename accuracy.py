
#Compute the accuracy of the clustering

def accuracy(cluster,y,K):
  """
  computes the accuracy of of the cluster
  input : clusters indexes, y : true labels 

  return accuracy : A =sum_k |{i ∈ Ck : i = mk}/number of elements
  """
  acc=0
  card=0
  
  for l in range(K):
    #find the mk : most frequent digit of cluster l
    mk= most_freq_digit(y[cluster[l]],K)
    print('The '+str(l+1)+'th cluster corresponds to the number '+str(mk))
    for element in y[cluster[l]]: 
      if int(element) == mk :
        acc+=1
      card +=1
  return acc/card #in fact card = len(y)


def most_freq_digit(liste,K):
  """
  find the most freq digit in liste, we suppose that liste contains int, from 0 to K-1
  """
  compteur=[0]*K #store how much time we see each element in the list
  for element in liste:
    element=int(element)
    compteur[element] += 1
  max_elem=max(compteur) 
  index=compteur.index(max_elem)
  return index
