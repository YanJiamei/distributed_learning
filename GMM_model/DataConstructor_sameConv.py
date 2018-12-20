import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.linalg import orth
import scipy.sparse
FEATURE_SIZE = 3
SPARSE_SIZE = 3
GMM_COMPONENT = 10
LABEL_NUM = 10
DATA_NUM = 1000
path = './newlabels/10.csv'
random.seed(1)
MAP_MATRIX = np.random.random_sample([FEATURE_SIZE, SPARSE_SIZE])
APPEND_LIST = np.random.random_sample([GMM_COMPONENT*DATA_NUM, FEATURE_SIZE])
APPEND_LABEL = np.random.randint(0,10,size = [GMM_COMPONENT*DATA_NUM,])
APPEND_gmm_id = np.random.randint(0,GMM_COMPONENT, size= [GMM_COMPONENT*DATA_NUM,])
print(APPEND_gmm_id)
print(APPEND_LABEL)
# SPARSE_MATRIX = scipy.sparse.rand(m = FEATURE_SIZE, n = SPARSE_SIZE, density=0.1)
# print(SPARSE_MATRIX)
def get_RndSymPosMatrix(size = FEATURE_SIZE):
    # random_list = []
    # start = 1
    # stop = 100
    # for i in range(size):
    #     random_list.append(random.randint(start, stop))
    # D = np.diag(np.array(random_list))
    D = np.diag(np.random.random_sample([size,]))/500
    V = np.random.rand(size, size)
    U = orth(V)
    D = mat(D)
    U = mat(U)
    A = U.I * D * U
    return A
def get_RndMean(size = FEATURE_SIZE):
    # random_list = []
    # start = 0
    # stop = 100
    # for i in range(size):
    #     random_list.append(random.randint(start, stop))
    random_list = np.random.random_sample([size,])
    return random_list

def data_construct(mean, conv,data_num = DATA_NUM, feature_size = FEATURE_SIZE, sparse_size = SPARSE_SIZE, gmm_size = GMM_COMPONENT):
    
    # mean = [get_RndMean() for i in range(gmm_size)]
    # conv = [get_RndSymPosMatrix() for i in range(gmm_size)]
    # assert isinstance(label, int),'label is not int'
    assert len(mean) == len(conv)
    count = len(mean)
    num = gmm_size * data_num
    x_sparse = np.empty(shape = [0,sparse_size+1])
    x = np.empty(shape = [0,feature_size+1])
    

    print('\nConstructing Gaussian Mixture Multivariate Data')
    for i in range(gmm_size):
        print(' (%d/%d)\tmean = %s, \n\tconvariance = %s' % (i+1,count,mean[i],conv[i]))
        gmm_id = np.ones((data_num,1)) * i
        temp = np.random.multivariate_normal(mean[i],conv[i],data_num)
        temp_m = np.matrix(temp)
        temp = np.concatenate((temp, gmm_id), axis = 1)
        temp_sparse = temp_m * MAP_MATRIX
        temp_sparse = np.concatenate((temp_sparse, gmm_id), axis = 1)
        x = np.concatenate((x, temp),axis = 0)
        x_sparse = np.concatenate((x_sparse, temp_sparse), axis = 0)
    temp = np.zeros(shape = [num,1])
    for i in range(num):
        k = x_sparse[i,0]
        p = x_sparse[i,1]
        if k < 0.45:
            if p < 0.45:
                _ = 0
            elif p < 0.55:
                _ = 1
            else: _ = 2
        elif k < 0.55: 
            if p < 0.45:
                _ = 3
            elif p < 0.55 :
                _ = 4
            else: _ = 5
        else:
            if p < 0.45:
                _ = 6
            elif p < 0.55:
                _ = 7
            else:
                _ = 8

        temp[i] = _
    temp = temp.astype(np.int64)
    x = np.round(x, decimals=4)
    x = np.concatenate((x, temp), axis = 1)
    x_sparse = np.round(x_sparse, decimals=4)
    x_sparse = np.concatenate((x_sparse, temp), axis = 1)
    dataframe = pd.DataFrame(data = x)
    dataframe_sparse = pd.DataFrame(data = x_sparse)
    # temp = pd.DataFrame([label for i in range(num)])
    # dataframe['label'] = temp
    dataframe.rename(columns={feature_size:'gmm_id', (feature_size+1):'label'}, inplace = True)
    dataframe_sparse.rename(columns={sparse_size:'gmm_id',(sparse_size+1):'label'}, inplace = True)
    return dataframe, dataframe_sparse

mean = [get_RndMean() for i in range(GMM_COMPONENT)]
conv = [get_RndSymPosMatrix() for i in range(GMM_COMPONENT)]

dataframe, dataframe_sparse = data_construct(mean = mean, conv = conv)
print(dataframe, dataframe_sparse)
dataframe.to_csv(path)
x = np.array(dataframe_sparse.loc[:,0])
y = np.array(dataframe_sparse.loc[:,1])
z = np.array(dataframe.loc[:,2])
# y = np.ones(shape = x.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, s=20, c='r')
plt.show()
x = np.array(dataframe .loc[:,0])
y = np.array(dataframe.loc[:,1])
z = np.array(dataframe.loc[:,2])
# y = np.ones(shape = x.shape)
fig1 = plt.figure()
ax = fig1.gca(projection='3d')
ax.scatter(x, y, z, s=20, c='g')
plt.show()

# dataframe_sparse = dataframe_sparse[0:4000]
# ## show 3D dots
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==0),0])
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==0),1])
# y = np.ones(shape = x.shape)
fig = plt.figure()
ax = fig.gca()
ax.scatter(x, y, s=20, c='r')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==1),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==1),1])
ax.scatter(x, y, s=20, c='b')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==2),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==2),1])
ax.scatter(x, y, s=20, c='g')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==3),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==3),1])
ax.scatter(x, y, s=20, c='k')
# x = np.array(dataframe_sparse.loc[(dataframe.gmm_id==4),0])
# y = np.array(dataframe_sparse.loc[(dataframe.gmm_id==4),1])
# ax.scatter(x, y, s=20, c='y')
# x = np.array(dataframe_sparse.loc[(dataframe.gmm_id==5),0])
# y = np.array(dataframe_sparse.loc[(dataframe.gmm_id==5),1])
# ax.scatter(x, y, s=20, c='m')
# x = np.array(dataframe_sparse.loc[(dataframe.gmm_id==6),0])
# y = np.array(dataframe_sparse.loc[(dataframe.gmm_id==6),1])
# ax.scatter(x, y, s=20, c='c')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==5),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==5),1])
ax.scatter(x, y, s=20, c='y')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==6),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==6),1])
ax.scatter(x, y, s=20, c='m')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==7),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==7),1])
ax.scatter(x, y, s=20, c='c')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==4),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==4),1])
ax.scatter(x, y, s=20, c='r')
x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==8),0])
# y = np.ones(shape = x.shape)
y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==8),1])
ax.scatter(x, y, s=20, c='g')
# x = np.array(dataframe_sparse.loc[(dataframe_sparse.label==7),0])
# # y = np.ones(shape = x.shape)
# y = np.array(dataframe_sparse.loc[(dataframe_sparse.label==7),1])
# ax.scatter(x, y, s=20, c='c')
plt.show()