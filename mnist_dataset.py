import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist as MDataset
import numpy as np
from tensorflow.python.framework import dtypes
import collections
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
WEIGHT_INIT = 1.#初始化所有来源样本的权重
WEIGHT_THRESHOLD = 1.2 #阻止节点之间的数据传输
DISTRIBUTE_NODE_NUM = 2 #参与的节点数目
TRANSFER_SIZE = 10 #相互传输的instance的数目，应该也是可以更改的

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

def extract_Dataset(datas,weights,labels):
    dataset = tf.data.Dataset.from_tensor_slices(({'x':datas,'w':weights},labels))
    dataset.shuffle(30000)
    return dataset
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
def load_data():
    mnist_datasets_load = np.load('./data/mnist_3node_datasets.npz') 
    mnist_datasets_load_even = mnist_datasets_load['mnist_even']
    mnist_datasets_load_odd = mnist_datasets_load['mnist_odd']
    mnist_datasets_load_mix = mnist_datasets_load['mnist_mix']
    mnist_odd = Datasets(train = mnist_datasets_load_odd[0], validation = mnist_datasets_load_odd[1], test = mnist_datasets_load_odd[2])
    mnist_even = Datasets(train = mnist_datasets_load_even[0], validation = mnist_datasets_load_even[1], test = mnist_datasets_load_even[2])   
    mnist_mix = Datasets(train = mnist_datasets_load_mix[0], validation = mnist_datasets_load_mix[1], test = mnist_datasets_load_mix[2])
    # weight = np.ones(shape = [train_labels.shape[0],],dtype=np.float32)

    mnist_odd_dataset = extract_Dataset(mnist_odd.train.images,
        weights = np.ones(shape = [mnist_odd.train.images.shape[0],],dtype=np.float32),
        labels = mnist_odd.train.labels )
    mnist_even_dataset = extract_Dataset(mnist_even.train.images,
        weights = np.ones(shape = [mnist_even.train.images.shape[0],],dtype=np.float32),
        labels = mnist_even.train.labels )
    mnist_mix_dataset = extract_Dataset(mnist_mix.train.images,
        weights = np.ones(shape = [mnist_mix.train.images.shape[0],],dtype=np.float32),
        labels = mnist_mix.train.labels )
    return mnist_odd_dataset, mnist_even_dataset, mnist_mix_dataset

datset_0, dataset_1, dataset_2 = load_data()
datset_0.batch(batch_size = BATCH_SIZE)
datset_0_iterator = datset_0.make_one_shot_iterator()
dataset_1.batch(batch_size = TRANSFER_SIZE)
dataset_1_iterator = dataset_1.make_one_shot_iterator()
dataset_2.batch(batch_size = TRANSFER_SIZE)
dataset_2_iterator = dataset_2.make_one_shot_iterator()
def inference(input_tensor, avg_class, weights1, biases1,
                weights2, biases2):
    if (avg_class == None):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return (tf.matmul(layer1, weights2) + biases2)
    else:
        pass

g0 = tf.Graph()
with g0.as_default():
 ##=====================Variable Initialization=================================
    #模型输入 shape = [None, Input]其中None表示batch_size大小
    # x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
    # y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
    # w = tf.placeholder(tf.float32, [None, 1], name = 'all-instance-weight-local')
    features_x_w, labels_y = datset_0_iterator.get_next()
    x = features_x_w['x']
    w = features_x_w['w']
    y = labels_y
    #to calculate the loss of a transferred BATCH, then adjust 'weights_node_1' for network
    x_transfer = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'transfer-x-input')
    y_transfer = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'transfer-y-input')
    #隐藏层参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1) #初始化
    )
    biases1 = tf.Variable(
        tf.constant(0.1, shape = [LAYER1_NODE])
    )

    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1)
    )
    biases2 = tf.Variable(
        tf.constant(0.1, shape = [OUTPUT_NODE])
    )
    #add weights_instance 表示各节点之间传输的数据的权重 一开始都为1
    #！！！权重不是乘在输入样本上 而是乘在loss函数中的！可增加部分样本对loss的影响！！！
    #权重w1=[w11,w12,...,w1n] 每次更新样本库都是更新w->w'
    #:可以将w和image、label组成数据结构，同步更新！
    weights_node_init = tf.Variable(
        tf.constant(WEIGHT_INIT, shape = [1], dtype=tf.float32), trainable = True
    )
 ##================================TRAINING=====================================
    #计算前向传播的结果
    y = inference(input_tensor = x, avg_class = None, weights1 = weights1
                    , biases1 = biases1, weights2 = weights2, biases2 = biases2)
    
    #存储训练轮数的变量
    global_step = tf.Variable(0, trainable = False)
    #add loss to update weights_instance
    
    #计算交叉熵——损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y, labels = tf.argmax(y_,1)
    )
    #给instance加权
    weighted_cross_entropy = tf.multiply(cross_entropy, tf.transpose(w))
    cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy)
    #L2正则化损失
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行，更新变量时使用的
                            # 学习率在这个基础上递减
        global_step,        # 当前迭代的轮数
        # mnist.train.num_examples / BATCH_SIZE, # 过完所有的训练数据需要的迭代次数
        1000,
        LEARNING_RATE_DECAY # 学习率的衰减速度
    )
    #优化损失函数loss
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step
                                                    , var_list=[weights1, biases1, weights2, biases2])
 ##=======================calculate accuracy====================================
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 注意这个accuracy是只跟average_y有关的，跟y是无关的
    # 这个运算首先讲一个布尔型的数值转化为实数型，然后计算平均值。这个平均值就是模型在这
    # 一组数据上的正确率 
    #用作validation和test——local
    correct_prediction_float = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction_float)
 ##=======================update weights_node_1===================================
    #本地加权accuracy_weighted_norm
    correct_prediction_float_weighted = tf.multiply(tf.transpose(w), correct_prediction_float)
    accuracy_weighted = tf.reduce_sum(correct_prediction_float_weighted)
    accuracy_weighted_norm = tf.divide(accuracy_weighted, tf.reduce_sum(w))
    #传输进来的一个batch的accuracy_transfer_batch
    y_transfer_infer = inference(input_tensor = x_transfer, avg_class = None, weights1 = weights1
                    , biases1 = biases1, weights2 = weights2, biases2 = biases2)
    correct_prediction_t = tf.equal(tf.argmax(y_transfer_infer, 1), tf.argmax(y_transfer, 1))
    correct_prediction_float_t = tf.cast(correct_prediction_t, tf.float32)
    accuracy_transfer_batch = tf.reduce_mean(correct_prediction_float_t)
    #计算weights_node
    exp_ = tf.exp(1.0-accuracy_transfer_batch)
    weights_node_update = (accuracy_weighted_norm)*exp_

    # loss_partition = 
with tf.Session(graph = g0) as sess:
    # 初始化变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 准备验证数据。一般在神经网络的训练过程中会通过验证数据要大致判断停止的
    # 条件和评判训练的效果。
    

    #INIT NET_WEIGHT
    weights_node_1 = sess.run(weights_node_init)
    weights_node_2 = sess.run(weights_node_init)


    #TRAINING BEGINS
    for i in range(TRAINING_STEPS):
        # 每1000轮输出一次在验证数据集上的测试结果
        
        # 产生这一轮使用的一个batch的训练数据，并运行训练过程。                  
        features, labels = datset_0_iterator.get_next()
        local_feed_0 = {
            x: features['x'], y_: labels, w:features['w']
        }
        sess.run(train_step, feed_dict = local_feed_0)
            

## "NEW TRANSFER" begins, add instance with weights to LOCAL DATA SET, add transferred instance without weights to TRANSFER POOL
            # xe, ye = mnist_datasets.train_even.next_batch(TRANSFER_SIZE)
        transfer_1_features, transfer_1_labels = dataset_1_iterator.get_next()
        # pool_1 = pool_1.concatenate(transfer_1)

        transfer_2_features, transfer_2_labels = dataset_2_iterator.get_next()
        # pool_1 = pool_1.concatenate(transfer_1)
##=========================================UPDATE weights_node_i============================================================           
            #待到传输的instance有一个batch之后 对比accuracy再做调整，这期间传输过来的数据不断地在更新训练，weights-node不变
        # if (pool_i == 0) and (i > 0):   
            # xt, yt = mnist_datasets.train_even.next_batch(BATCH_SIZE)
        features, labels = datset_0_iterator.get_next()
        weights_update_feed_1 = {
            x_transfer: transfer_1_features['x'],
            y_transfer: transfer_1_labels,
            x :features['x'], 
            y_:labels, 
            w: features['w']
        }
        weights_update_feed_2 = {
            x_transfer: transfer_2_features['x'],
            y_transfer: transfer_2_labels,
            x :features['x'], 
            y_:labels, 
            w: features['w']
        }
        weights_node_1 = sess.run(weights_node_update, feed_dict=weights_update_feed_1 )
        weights_node_2 = sess.run(weights_node_update, feed_dict=weights_update_feed_2 )
        weightset_1 = extract_Dataset(transfer_1_features['x'],weights_node_1,transfer_1_labels)
        weightset_2 = extract_Dataset(transfer_2_features['x'],weights_node_2,transfer_2_labels)
##=========================================RECEIVE & SHUFFLE ALL DATA======================================================
        if weights_node_1 > WEIGHT_THRESHOLD:
            dataset_0 = dataset_0.concatenate(weightset_1)
        if weights_node_2 > WEIGHT_THRESHOLD:
            dataset_0 = dataset_0.concatenate(weightset_2)
        if (weights_node_1 > WEIGHT_THRESHOLD) or (weights_node_2 > WEIGHT_THRESHOLD):
            dataset_0.shuffle(3000)

        if i % 1 == 0:
            # 计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次
            # 可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更
            # 小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch
            # 会导致计算时间过长甚至发生内存溢出的错误。
            # 注意我们用的是滑动平均之后的模型来跑我们验证集的accuracy
            print("===============TRANSFER WEIGHT: w1 = %f , w2 = %f================ \n" % (weights_node_1, weights_node_2))

            validate_acc = sess.run(accuracy, feed_dict=local_feed_0)
            print("After %d training step(s), validation accuracy-all is %g " % (i, validate_acc))
            # validate_acc_odd = sess.run(accuracy, feed_dict = validate_feed_0)
            # print("After %d training step(s), validation accuracy-odd is %g " % (i, validate_acc_odd))
            # validate_acc_odd = sess.run(accuracy, feed_dict = validate_feed_1)
            # print("After %d training step(s), validation accuracy-even is %g \n" % (i, validate_acc_odd))
##=========================================NEXT TRAINING===================================================================       
        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        # 同样，我们最终的模型用的是滑动平均之后的模型，从这个accuracy函数
        # 的调用就可以看出来了，因为accuracy只与average_y有关
    # test_acc = sess.run(accuracy, feed_dict=test_feed)
    # print("After %d training step(s), test accuracy-all is %g" % (TRAINING_STEPS, test_acc))
    # test_acc_odd = sess.run(accuracy,feed_dict=test_feed_0)
    # print("After %d training step(s), test accuracy-odd is %g" % (TRAINING_STEPS, test_acc_odd))
    # test_acc_odd = sess.run(accuracy,feed_dict=test_feed_1)
    # print("After %d training step(s), test accuracy-even is %g\n=============================================================\n" % (TRAINING_STEPS, test_acc_odd))

# 主程序入口

# def main(argv=None):
#     # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
#     # mnist = input_data.read_data_sets("./data", one_hot=True)
#     datset_0, dataset_1, dataset_2 = load_data()
#     # train(mnist=mnist, datanode_0 = mnist_odd, datanode_1 = mnist_even, datanode_2 = mnist_mix)
#     print('ok')
# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
# if __name__ == "__main__":
#     tf.app.run()
