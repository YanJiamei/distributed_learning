import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.linalg import orth
import scipy.sparse
FEATURE_SIZE = 10
SPARSE_SIZE = 50
sparse_flag = True
GMM_COMPONENT = 10
LABEL_NUM = 10
DATA_NUM = 10000
GMM_COMPO_NUM = 100
path = './origin/feature10-50gmm10.csv'
def feature_name(label, size=FEATURE_SIZE):
    return [str(label)+'_'+str(x) for x in range(size)]
# MAP_MATRIX = np.random.random_sample([FEATURE_SIZE, SPARSE_SIZE])
MAP_MATRIX = np.random.randint(0,2,size=(FEATURE_SIZE, SPARSE_SIZE))
# SPARSE_MATRIX = scipy.sparse.rand(m = FEATURE_SIZE, n = SPARSE_SIZE, density=0.1)
# print(SPARSE_MATRIX)
def get_RndSymPosMatrix(size = FEATURE_SIZE):
    random_list = []
    start = 1
    stop = 100
    for i in range(size):
        random_list.append(random.randint(start, stop))
    D = np.diag(np.array(random_list))
    V = np.random.rand(size, size)
    U = orth(V)
    D = mat(D)
    U = mat(U)
    A = U.I * D * U
    return A
def get_RndMean(size = FEATURE_SIZE):
    random_list = []
    start = 1
    stop = 100
    for i in range(size):
        random_list.append(random.randint(start, stop))
    return random_list

# invariant_conv = [get_RndSymPosMatrix(size=FEATURE_SIZE) for i in range(GMM_COMPONENT)]

def data_construct(label, num, gmm_id, size = FEATURE_SIZE, gmm_size = 1):
    
    mean = [get_RndMean() for i in range(gmm_size)]
    conv = [get_RndSymPosMatrix() for i in range(gmm_size)]
    assert isinstance(label, int),'label is not int'
    assert len(mean) == len(conv)
    count = len(mean)

    if sparse_flag:
        x = np.empty(shape = [0,SPARSE_SIZE])
    else:
        x = np.empty(shape = [0,FEATURE_SIZE])
    

    print('\nConstructing Gaussian Mixture Multivariate Dataof label -%d-:'%label)
    for i in range(count):
        print(' (%d/%d)\tmean = %s, \n\tconvariance = %s' % (i+1,count,mean[i],conv[i]))
        temp = np.random.multivariate_normal(mean[i],conv[i],num//gmm_size)
        if sparse_flag == True:
            temp_m = np.matrix(temp)
            temp = temp_m * MAP_MATRIX
        x = np.concatenate((x, temp),axis = 0)

    x = np.round(x, decimals=4)
    dataframe = pd.DataFrame(data = x)
    temp = pd.DataFrame([label for i in range(num)])
    dataframe['label'] = temp
    temp = pd.DataFrame([gmm_id for i in range(num)])
    dataframe['gmm_id'] = temp
    return dataframe, (mean, conv)

dataframe = pd.DataFrame()
for j in range(GMM_COMPONENT):
    for i in range(LABEL_NUM):
        dataframe_temp, _ = data_construct(label = i,gmm_id = j, num = GMM_COMPO_NUM)
        dataframe = pd.concat((dataframe,dataframe_temp), axis=0, ignore_index=True)
# for i in range(LABEL_NUM):
#         dataframe_temp, _ = data_construct(label = i,gmm_id = 0, num = DATA_NUM, size = FEATURE_SIZE, gmm_size = GMM_COMPONENT)
#         dataframe = pd.concat((dataframe,dataframe_temp), axis=0, ignore_index=True)
dataframe.to_csv(path)

# dataframe_A,_ = data_construct(label = 0, num = DATA_NUM)
# dataframe_B,_ = data_construct(label = 1, num = DATA_NUM)
# dataframe_C,_ = data_construct(label = 2, num = DATA_NUM)
# dataframe = pd.concat((dataframe_A,dataframe_B,dataframe_C),axis=0,ignore_index=True)
# dataframe.to_csv('./GMM_model/GMM_data/G_3M_4M_10000.csv')
print(dataframe)
# print(dataframe)

# ## show 3D dots
x = np.array(dataframe.loc[(dataframe.label==1),0])
y = np.array(dataframe.loc[(dataframe.label==1),1])
z = np.array(dataframe.loc[(dataframe.label==1),2])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, s=20, c='r', depthshade=True)
x = np.array(dataframe.loc[(dataframe.label==2),0])
y = np.array(dataframe.loc[(dataframe.label==2),1])
z = np.array(dataframe.loc[(dataframe.label==2),2])
ax.scatter(x, y, z, s=20, c='b', depthshade=True)
x = np.array(dataframe.loc[(dataframe.label==0),0])
y = np.array(dataframe.loc[(dataframe.label==0),1])
z = np.array(dataframe.loc[(dataframe.label==0),2])
ax.scatter(x, y, z, s=20, c='g', depthshade=True)
x = np.array(dataframe.loc[(dataframe.label==3),0])
y = np.array(dataframe.loc[(dataframe.label==3),1])
z = np.array(dataframe.loc[(dataframe.label==3),2])
ax.scatter(x, y, z, s=20, c='k', depthshade=True)
x = np.array(dataframe.loc[(dataframe.label==4),0])
y = np.array(dataframe.loc[(dataframe.label==4),1])
z = np.array(dataframe.loc[(dataframe.label==4),2])
ax.scatter(x, y, z, s=20, c='y', depthshade=True)
x = np.array(dataframe.loc[(dataframe.label==5),0])
y = np.array(dataframe.loc[(dataframe.label==5),1])
z = np.array(dataframe.loc[(dataframe.label==5),2])
ax.scatter(x, y, z, s=20, c='m', depthshade=True)
x = np.array(dataframe.loc[(dataframe.label==6),0])
y = np.array(dataframe.loc[(dataframe.label==6),1])
z = np.array(dataframe.loc[(dataframe.label==6),2])
ax.scatter(x, y, z, s=20, c='c', depthshade=True)
plt.show()
# ## show 3D dots
# x = np.array(dataframe.loc[(dataframe.label==1) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==1) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==1) & (dataframe.gmm_id==0),2])
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, s=20, c='r', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==2) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==2) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==2) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='b', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==0) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==0) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==0) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='g', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==3) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==3) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==3) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='k', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==4) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==4) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==4) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='y', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==5) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==5) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==5) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='m', depthshade=True)
# x = np.array(dataframe.loc[(dataframe.label==6) & (dataframe.gmm_id==0),0])
# y = np.array(dataframe.loc[(dataframe.label==6) & (dataframe.gmm_id==0),1])
# z = np.array(dataframe.loc[(dataframe.label==6) & (dataframe.gmm_id==0),2])
# ax.scatter(x, y, z, s=20, c='c', depthshade=True)
# plt.show()