# SS_DBSCAN
A developed version of DBSCAN (Semi-Supervised Clustering)

To run the code you need to pass 4 arguments ( dataset as csv with last column as the class. eps value, MinPts value and a text file name to save the classes).
#run call example :python3 SSDBSCAN.py  lettersPreProc.csv 8 17 classes.txt

For testing we provided the dataset csv file "lettersPreProc.csv"  that is based on Letters dataset (P. W. Frey and D. J. Slate, ‘‘Letter recognition using holland-style adaptive
classifiers,’’ Machine learning, vol. 6, pp. 161–182, 1991.) with slightly preprocessing where the letter is mapped to integer and placed as the last column (column 17). Columns (1-16) are the features values.

The output is the classes.txt file, and the code will displays the performance of the clustering using V-measure, Silhouette, and ARI metrics.
