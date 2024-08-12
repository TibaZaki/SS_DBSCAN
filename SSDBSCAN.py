

#This code depends on the code Copyright (c) 2016 chrisjmccormick, and thus is following same license Copyright (c) 2020 Tiba.Zaki
#https://github.com/chrisjmccormick/dbscan/blob/master/LICENSE
#Author: Tiba Zaki Abdulhameed Feb 24,2020, Al-Nahrain University and Western Michigan University
#!/usr/bin/python3
# -*- coding: utf-8 -*- 
#run call example :python3 SSDBSCAN.py  lettersPreProc.csv 8 17 classes.txt
 
#--------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd


import sys 
import codecs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


import scipy
from scipy.spatial import distance
core_counter = 0
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import davies_bouldin_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score

def SSDBSCAN( eps, MinPts,Min,Max,Mode,Features,M):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    SSDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
 
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0]*len(Features)
    # C is the ID of the current cluster.    
    C = 0
    vocab_size=len(Features)
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)

    for P in range(0, vocab_size):
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
           continue
        
        # Find all of P's neighboring points.
        NeighborPts = regionQuery( P, eps,M,vocab_size)
        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           C += 1
           growCluster( labels, P, NeighborPts, C, eps, MinPts,Min,Max,Mode,Features,M,vocab_size)
    
    # All data has been clustered!
    return labels


def growCluster( labels, P, NeighborPts, C, eps, MinPts,Min,Max,Mode,Features,distance_Matrix,vocab_size):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[P] = C
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    global core_counter
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
           labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            
            # Find all the neighbors of Pn
            PnNeighborPts = regionQuery( Pn, eps,distance_Matrix,vocab_size)
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            core_condition=Is_important(Pn,Min,Max,Mode,Features[Pn])
            
            #print(core_condition)
            #print(Min,Max,Features[Pn])
            is_core=len(PnNeighborPts) >= MinPts and core_condition
            
            if is_core:
                NeighborPts = NeighborPts + PnNeighborPts
                core_counter +=1
                #print(core_counter)
                
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts               
        
        # Advance to the next point in the FIFO queue.
        i += 1        

    # We've finished growing cluster C!
def Is_important(Pn,Min,Max,Mode,attrs):
    flag = False
    count=0
    far=0
    near=0

    for i in  6,7,8,9,11,12,13,14,15,16: # the selected features
    
        if (attrs[i-1]>=Max[i-1]-2): # This condition has been tested in our article, but you can change it depending on your dataset and clustering purpose.
            near=near+1 
            
    if far>=1 or near>=1: flag=True       
    return(flag)			
		 



def regionQuery(P, eps,Distance_matrix,N):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, N):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if Distance_matrix[P][Pn] < eps:
            neighbors.append(Pn)	
	#D2=scipy.spatial.distance.euclidean(model[w2], model[word])
        #if numpy.linalg.norm(D[P] - D[Pn]) < eps:
        #   neighbors.append(Pn)
            
    return neighbors

#-----------------------------------------------------------------------------------------------------
def pre_process(df):
  
  X = df.values[:, :-1]
  target = df.values[:, -1]

  
  return X, target
#-----------------------------------------------------------------------------------------------------
def SaveCLustersToFile(c,f):



  df = pd.DataFrame(c)

  # Save to CSV
  df.to_csv(f, index=False)

def main():
    
    if(len(sys.argv)!=5):print('Arguments error ') # output classes file, input eps value
    else:
        CSVDataFile=sys.argv[1]
        df = pd.read_csv(CSVDataFile, header=None)
        X, target = pre_process(df)# Assumes lat column is the class(target)
        
        OriginalClasses=target
        Data_size= len(X)       

        
        Distance_matrix = [[0 for x in range(Data_size)] for y in range(Data_size)] 
        Distance_matrix = distance.cdist(X, X, 'euclidean')
        print(X)
#--------------------------------------------------------
        #print('Start clustering')	
        # Tiba DBSCAN V2
        eps=float(sys.argv[2]) # Suhad
        print("The eps in SSDBSCAN is",eps)
        
        min_samples=float(sys.argv[3])
        print("The MinPoints=",min_samples)
        ClutersFile=sys.argv[4]
        print("Output clusters file is ", ClutersFile)
   
             
     
#------------------------------------------------------
        #Start Clustering Original DBSCAN to compare
        clustering=DBSCAN(eps=eps, min_samples=min_samples).fit(X)	
        clustersDB =clustering.labels_
        
#------------------------------------------------------
        # The Max, Avrg, and Mode are previously extracted from the dataset. These values can be tested to decide the Is-Important condition. In our case we used
        # the just the Max value.
	Max=[15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,	15,]
        Avrg=[	4.02355,	7.0355,	5.12185,	5.37245,	3.50585,	6.8976,	7.50045,	4.6286,	5.17865,	8.28205,	6.454,	7.929,	3.0461,	8.33885,	3.69175,	7.8012]
        Mode=[-59,	-56,	-56,	-59,	-64,	-87,	-85] 
        Data_size=len(X)
        print("Data size= ",Data_size)         
        clusters= SSDBSCAN(eps, min_samples,Avrg,Max,Mode,X,Distance_matrix)
        
        SaveCLustersToFile(clusters,ClutersFile)
#---------------------------------------------------------
        import hdbscan
        
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        clusterer.fit(X)
        clustersH =clusterer.labels_
#----------------------------------------------------
       
	line=str(eps)+', HDBSCAN silhouette_score ,'+str(silhouette_score(X, clustersH ))
	line=line+'\n'+'HDBSCAN vmeasure ,'+str(v_measure_score(OriginalClasses, clustersH ))
        line=line+'\n'+'HDBSCAN ARI'+str(adjusted_rand_score(OriginalClasses,clustersH))
	line=line+'\n-------------------'
        line=line+'\n'+str(eps)+', DBSCAN silhouette_score ,'+str(silhouette_score(X, clustersDB )) 
	line=line+'\n'+'DBSCAN v_measure ,'+str(v_measure_score(OriginalClasses, clustersDB ))    
        line=line+'\n'+'DBSCAN ARI'+str(adjusted_rand_score(OriginalClasses,clustersDB))
        line=line+'\n-------------------'
        line=line+'\n'+str(eps)+', SSDBSCAN silhouette_score ,'+str(silhouette_score(X, clusters )) 
	line=line+'\n'+'SSDBSCAN v_measure ,'+str(v_measure_score(OriginalClasses, clusters ))    
        line=line+'\n'+'SSDBSCAN ARI'+str(adjusted_rand_score(OriginalClasses,clusters))
        line=line+'\n-------------------'
       
        print (line)
        

    

main()
