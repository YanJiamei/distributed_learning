import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feature_name(size, label):
    return [str(label)+'_'+str(x) for x in range(size)]

def data_construct(mean, conv, label, num):
    assert isinstance(label, str),'label is not string'
    assert len(mean) == len(conv)
    count = len(mean)
    x = 0
    for i in range(count):
        x = x + np.random.multivariate_normal(mean[i],conv[i],num)
    x = np.round(x, decimals=4)
    a,b = x.T
    plt.scatter(a,b)
    plt.show()
    dataframe = pd.DataFrame(data = x)
    new_col = feature_name(dataframe.shape[1], label)
    dataframe.columns = new_col
    return dataframe

mean_A_1 = [-100,-100]
conv_A_1 = [[1,6],[6,10]]
mean_A_2 = [100,100]
conv_A_2 = [[1,2],[2,10]]
mean = [mean_A_1, mean_A_2]
conv = [conv_A_1, conv_A_2]
dataframe = data_construct(mean = mean, conv = conv, label = 'A', num = 100)

# dataframe_A_2 = data_construct(mean = mean_A_1, conv = conv_A_1, label = 'A', num = 100)
print(dataframe)
