import numpy as np
from pandas.core.accessor import register_dataframe_accessor
import scipy as sp
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats.stats import _compute_dplus 


# initialization function for k means++
def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    
    init_centers = []
    # random initial center
    r = np.random.choice(len(X))
    r_val = X.iloc[r,:]
    
    init_centers.append(r_val)
    if(k == 1):
        return init_centers

    # print(init_centers)
    


    for i in range(k-1):
        dist = []
        probs = []
        #  find min distance of each point to each center
        for j in range(len(X)):
            t = []
            for k in range(len(init_centers)):
                temp = np.linalg.norm(init_centers[k]- X.iloc[j,:])**2
                t.append(temp)

        # get all min dist
            d = min(t)
            dist.append(d)
            
            
        #  calcualte probabilites
        probs = [dist[l]/sum(dist) for l in range(len(X))]
        indices = [i for i in range(len(X))]
        #  choose random centers based on probabilities
        new_center = np.random.choice(indices, 1, probs)
        new_center_data = X.iloc[new_center,:]
        init_centers = np.vstack((init_centers, new_center_data))
        
    return(init_centers)

    


    

#  apply k means algo
def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    #  get initial centers 
    final_centers = k_init(X, k)

    # clustering objective 
    c_obj = compute_objective(X, final_centers)


    # iters = [i+1 for i in range(max_iter)]
    # plt_c_obj = []


    for i in range(max_iter):
        sums_list = np.zeros([k, X.shape[1]])
        cts = np.zeros(k)
        data_map = assign_data2clusters(X, final_centers)
        
        for j in range(len(X)):
            for l in range(k):
                if(data_map[j][l] == 1):
                    sums_list[l] += X.iloc[j,:]
                    cts[l] += 1

        
        new_centers = [sums_list[j]/ int(cts[j]) for j in range(k)]
        new_centers = np.array(new_centers)
        
        new_c_obj = compute_objective(X, new_centers)
        if(new_c_obj < c_obj):
            c_obj = new_c_obj
            final_centers = new_centers

        
    #     plt_c_obj.append(c_obj)

    # plt.plot(iters, plt_c_obj)
    # plt.show()
    return final_centers
        
        





def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    data_map = np.zeros([len(X), len(C)])
    # print(data_map.shape)
    len_x = len(X)
    len_c = len(C)
    dist = []


    for i in range(len_x):
        min_d = np.inf
        req_c = -1
        # find eucledian distances 
        for j in range(len_c):
            # print(X.iloc[i,:])
            temp = np.linalg.norm(C[j] - X.iloc[i,:])**2
            # find min dist out of for given point
            if(temp < min_d):
                min_d = temp
                req_c = j
            # dist[i][j] = temp
        
        # print(dist[i])
        # assign correct cluster to datapoint 
        #  create binary matrix
        for i in range(len(C)):
            if(i == req_c):
                data_map[i] = 1
            else:
                data_map[i] = 0


    return data_map




# compute objective for every assignments
def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    
    accuracy = 0
    len_x = len(X)
    len_c = len(C)
    for i in range(len_x):
        dist_min = np.inf

        # calcualte eucledian distance 
        for j in range(1, len_c):
            temp = np.linalg.norm(C[j]- X.iloc[i,:])**2
            if temp < dist_min:
                dist_min = temp
        # sum the min distances to get accuracy
        accuracy = accuracy + dist_min

    return accuracy


    

# plotting of graphs 
# TODO
data = pd.read_csv("iris.data")

data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
# print(data.head)
x1 = data['sepal length']/data['sepal width']
x2 = data['petal length']/data['petal width']
y = data['class']

data_new = pd.DataFrame()
data_new['x1'] = x1
data_new['x2'] = x2
# data_new['y'] = y

# x1 vs x2
# for name, group in data.groupby("class"):
#     plt.scatter(x1[group.index], x2[group.index], label=name)

# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()



# Run algorithm 50 times over the data with different values of clustersk= 1,2,...,5 and plot the accuracies
# acc = []
# centers = k_means_pp(data_new, 1, 50)
# acc.append(compute_objective(data_new, centers))

# centers = k_means_pp(data_new, 2, 50)
# acc.append(compute_objective(data_new, centers))

# centers = k_means_pp(data_new, 3, 50)
# acc.append(compute_objective(data_new, centers))

# centers = k_means_pp(data_new, 4, 50)
# acc.append(compute_objective(data_new, centers))

# centers = k_means_pp(data_new, 5, 50)
# acc.append(compute_objective(data_new, centers))

# plt.plot([1, 2, 3, 4, 5], acc)
# plt.xlabel("clusters")
# plt.ylabel("Objective")
# plt.show()

# best result is with 3 clusters 
# f_centers = k_means_pp(data_new, 3, 50)

# for center in f_centers:
#     for name, group in data.groupby("class"):
#            plt.scatter(x1[group.index], x2[group.index], label=name)
    
#     plt.scatter(center[0], center[1], c='red', s=300)
# plt.show()


