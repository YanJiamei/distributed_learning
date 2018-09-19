from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist 
import numpy as np
from tensorflow.python.framework import dtypes
import collections
def extract_n_data_sets(datasets, label = [1,2,3]):
    test_data_set_image = datasets.images
    test_data_set_label = datasets.labels
    extract_images = np.array([])
    extract_labels = np.array([])
    cnt = 0
    for i in range(datasets.num_examples):
        if (np.argmax (test_data_set_label[i]) in label):
            cnt += 1
            extract_images = np.append(extract_images , test_data_set_image[i])
            extract_labels = np.append(extract_labels , test_data_set_label[i])
    extract_images = extract_images.astype(np.float32)
        # return mnist.DataSet(extract_images, extract_labels, dtype = dtypes.float32, reshape = True)
    return mnist.DataSet(extract_images.reshape(cnt,784), extract_labels.reshape(cnt,10)
                        , dtype = dtypes.uint8, reshape = False)

Datasets = collections.namedtuple('Datasets', ['train_odd', 'train_even'
                                    , 'test_odd', 'test_even', 'validation_odd', 'validation_even'])

def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    mnist = input_data.read_data_sets("./data", one_hot=True)
    # train(mnist)
    mnist_13579_train = extract_n_data_sets(mnist.train,label=[1,3,5,7,9])
    mnist_24680_train = extract_n_data_sets(mnist.train,label=[2,4,6,8,0])
    mnist_13579_validation = extract_n_data_sets(mnist.validation, label=[1,3,5,7,9])
    mnist_24680_validation = extract_n_data_sets(mnist.validation, label=[2,4,6,8,0])
    mnist_13579_test = extract_n_data_sets(mnist.test, label=[1,3,5,7,9])
    mnist_24680_test = extract_n_data_sets(mnist.test, label=[2,4,6,8,0])
    mnist_datasets = Datasets(train_odd = mnist_13579_train, train_even = mnist_24680_train
                                    , validation_odd = mnist_13579_validation, validation_even = mnist_24680_validation
                                    , test_odd = mnist_13579_test, test_even = mnist_24680_test)
    np.save('./data/mnist_datasets.npy', mnist_datasets)
    # x1,y1 = extract_n_data_sets(mnist,label=1)
    print ('ok')

# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    main()