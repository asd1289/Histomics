import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale 
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import scipy.io as sio 

# Extracting features and saving as .csv file from.mat file

os.chdir("/gpfs/commons/home/deshpandea-934/data/Histomics_LUAD")
tcga_files = glob.glob("TCGA*")
for file in tcga_files:
        mat_file = sio.loadmat(file)
        df = pd.DataFrame(mat_file['Features'])# Extracting the features
        df.to_csv(file + '.csv', index = False)

# Creatin a master file with features from all samples.

os.chdir("/gpfs/commons/home/deshpandea-934/data/Histomics_LUAD/csv_featurs")
csv_list = glob.glob('*DX1*')
all_406 = []
for file in csv_list:
    df = pd.read_csv(file)
    all_406.append(df)
conc = pd.concat(all_406)
conc.to_csv("all_406.csv", index = False)

# Next subsample every hundreth row from this massive file

f = "all_406.csv"
n = 100 #The nth row to select
num_lines = 264222102#total numnber of rows in file
skip = [x for x in range(1, num_lines) if x %n != 0]#Making list of all other rows
df1 = pd.read_csv(f, skiprows = skip)#Using skip to skip rows
df1.to_csv("100_subsample.csv", index = False)

# Used R to get within group sum of squres for each k. Optimum k at 14

# After examining the plot the optimum k == 14
# Using this subsample and optimum k, fitted the model using minibatchkmeans in python
df = pd.read_csv("100_subsample.csv")
X = df.as_matrix()
X = scale(X)
clusterer = MiniBatchKMeans(n_clusters = 14, max_iter = 1000, batch_size = 1000, n_init = 25)
clusterer.fit(X)
centers = clusterer.cluster_centers_

# Once the cluster centers were obtained, used these centers to get labels for each cell as follows:
os.chdir("/gpfs/commons/home/deshpandea-934/data/Histomics_LUAD") 
mat_file = glob.glob("*DX1*") 
prop = []

##Wrote following for loop to:
for file in mat_file:
    f = sio.loadmat(file)#load each sample
    d = f['Features']#Extract features
    d = d[:,2:]#remove unwanted columns with magnification information
    d = scale(d)#scale data
    l = pairwise_distances_argmin(d, centers) #obtain label for each cell based on min(euclidean distance from centers)
    l = l.reshape(len(l),1)#reshape labels
    x = f['cX']#get x coordinate
    y = f['cY']#get y coordinate
    a = np.concatenate((x,y,l), axis = 1)#concat x,y and label used to plot later
    a = pd.DataFrame(a)#convert to pandas df
    a.to_csv(file + 'labels.csv', index = False)#write each file for each sample
    s = pd.DataFrame(a[2].value_counts()/len(a))#get proportions for each cluster type per sample
    s['cluster'] = s.index#inex info contained cluster type, included here
    s['sample'] = file#file name
    prop.append(s)#appended the list of df

conc = pd.concat(prop)
conc.to_csv("All_samples_prop.csv", index = False)

