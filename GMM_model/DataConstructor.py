import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.linalg import orth
FEATURE_SIZE = 10
GMM_COMPONENT = 4
LABEL_NUM = 10
DATA_NUM = 100000
def feature_name(label, size=FEATURE_SIZE):
    return [str(label)+'_'+str(x) for x in range(size)]


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
    start = 0
    stop = 100
    for i in range(size):
        random_list.append(random.randint(start, stop))
    return random_list

def data_construct(label, num, size = FEATURE_SIZE, gmm_size = GMM_COMPONENT):
    mean = [get_RndMean() for i in range(gmm_size)]
    conv = [get_RndSymPosMatrix() for i in range(gmm_size)]
    assert isinstance(label, int),'label is not int'
    assert len(mean) == len(conv)
    count = len(mean)
    x = np.empty(shape = [0,size])
    print('\nConstructing Gaussian Mixture Multivariate Dataof label -%d-:'%label)
    for i in range(count):
        print(' (%d/%d)\tmean = %s, \n\tconvariance = %s' % (i+1,count,mean[i],conv[i]))
        temp = np.random.multivariate_normal(mean[i],conv[i],num//gmm_size)
        x = np.concatenate((x, temp),axis = 0)
    x = np.round(x, decimals=4)
    # a,b = x.T
    # plt.scatter(a,b)
    # plt.show()
    dataframe = pd.DataFrame(data = x)
    temp = pd.DataFrame([label for i in range(num)])
    dataframe['label'] = temp
    # new_col = feature_name(label)
    # dataframe.columns = new_col
    # dataframe = pd.concat((dataframe,label),axis=1)
    return dataframe, (mean, conv)

dataframe = pd.DataFrame()
for i in range(LABEL_NUM):
    dataframe_temp, _ = data_construct(label = i, num = DATA_NUM)
    dataframe = pd.concat((dataframe,dataframe_temp), axis=0, ignore_index=True)
dataframe.to_csv('./GMM_model/GMM_data/G_10M_4M_100000.csv')

# dataframe_A,_ = data_construct(label = 0, num = DATA_NUM)
# dataframe_B,_ = data_construct(label = 1, num = DATA_NUM)
# dataframe_C,_ = data_construct(label = 2, num = DATA_NUM)
# dataframe = pd.concat((dataframe_A,dataframe_B,dataframe_C),axis=0,ignore_index=True)
# dataframe.to_csv('./GMM_model/GMM_data/G_3M_4M_10000.csv')
print(dataframe)
# print(dataframe)

# ## show 3D dots
# x = np.array(dataframe.loc[dataframe.label==1,0])
# y = np.array(dataframe.loc[dataframe.label==1,1])
# z = np.array(dataframe.loc[dataframe.label==1,2])
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(x, y, z, s=20, c='r', depthshade=True)
# x = np.array(dataframe.loc[dataframe.label==2,0])
# y = np.array(dataframe.loc[dataframe.label==2,1])
# z = np.array(dataframe.loc[dataframe.label==2,2])
# ax.scatter(x, y, z, s=20, c='b', depthshade=True)
# x = np.array(dataframe.loc[dataframe.label==3,0])
# y = np.array(dataframe.loc[dataframe.label==3,1])
# z = np.array(dataframe.loc[dataframe.label==3,2])
# ax.scatter(x, y, z, s=20, c='g', depthshade=True)
# plt.show()