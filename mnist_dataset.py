import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist as MDataset
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes
import collections
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
WEIGHT_INIT = 1.#初始化所有来源样本的权重
WEIGHT_THRESHOLD = 1.0 #阻止节点之间的数据传输
DISTRIBUTE_NODE_NUM = 2 #参与的节点数目
TRANSFER_FRE = 10000 #相互传输的instance的频率，应该也是可以更改的

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
log_dir = "./logs/no_transfer/"
# def extract_Dataset(datas,weights,labels):
#     dataset = tf.data.Dataset.from_tensor_slices((datas,weights,labels))
#     dataset.shuffle(1000)
#     return dataset
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def extract_DF(datas,labels,weights):
    data = pd.DataFrame(datas)
    label = pd.DataFrame(labels)
    weight = pd.DataFrame(weights)
    frames = [data,label,weight]
    df = pd.concat(frames,axis=1,ignore_index=True)
    return df
    

def load_data():
    mnist = input_data.read_data_sets("./data", one_hot=True)
    mnist_datasets_load = np.load('./data/mnist_3node_datasets.npz') 
    mnist_datasets_load_even = mnist_datasets_load['mnist_even']
    mnist_datasets_load_odd = mnist_datasets_load['mnist_odd']
    mnist_datasets_load_mix = mnist_datasets_load['mnist_mix']
    mnist_odd = Datasets(train = mnist_datasets_load_odd[0], validation = mnist_datasets_load_odd[1], test = mnist_datasets_load_odd[2])
    mnist_even = Datasets(train = mnist_datasets_load_even[0], validation = mnist_datasets_load_even[1], test = mnist_datasets_load_even[2])   
    mnist_mix = Datasets(train = mnist_datasets_load_mix[0], validation = mnist_datasets_load_mix[1], test = mnist_datasets_load_mix[2])
    # weight = np.ones(shape = [train_labels.shape[0],],dtype=np.float32)

    mnist_odd_dataset = extract_DF(datas=mnist_odd.train.images,
        weights = np.ones(shape = [mnist_odd.train.images.shape[0],],dtype=np.float32),
        labels = mnist_odd.train.labels )
    mnist_even_dataset = extract_DF(datas=mnist_even.train.images,
        weights = np.ones(shape = [mnist_even.train.images.shape[0],],dtype=np.float32),
        labels = mnist_even.train.labels )
    mnist_mix_dataset = extract_DF(datas=mnist_mix.train.images,
        weights = np.ones(shape = [mnist_mix.train.images.shape[0],],dtype=np.float32),
        labels = mnist_mix.train.labels )
    mnist_all_test = extract_DF(datas=mnist.test.images,
        labels = mnist.test.labels,
        weights = np.ones(shape = [mnist.test.labels.shape[0],]))
    mnist_all_train = extract_DF(datas=mnist.train.images,
        labels = mnist.train.labels,
        weights = np.ones(shape = [mnist.train.labels.shape[0],]))
    return mnist_odd_dataset, mnist_even_dataset, mnist_mix_dataset, mnist_all_test, mnist_all_train


def inference(input_tensor, avg_class, weights1, biases1,
                weights2, biases2):
    if (avg_class == None):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return (tf.matmul(layer1, weights2) + biases2)
    else:
        pass

g0 = tf.Graph()
g1 = tf.Graph()
with g0.as_default():
 ##=====================Variable Initialization=================================
    #模型输入 shape = [None, Input]其中None表示batch_size大小
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
        w = tf.placeholder(tf.float32, [None, ], name = 'all-instance-weight-local')
    #to calculate the loss of a transferred BATCH, then adjust 'weights_node_1' for network
    with tf.name_scope('transfer_data'):
        x_transfer = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'transfer-x-input')
        y_transfer = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'transfer-y-input')

    #隐藏层参数
    with tf.name_scope('parameters'):
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
    with tf.name_scope('transfer_node_weight'):
        weights_node_init = tf.Variable(
            tf.constant(WEIGHT_INIT, shape = [1], dtype=tf.float32), trainable = True
        )
 ##================================TRAINING=====================================
    #计算前向传播的结果
    with tf.name_scope('inference'):
        y = inference(input_tensor = x, avg_class = None, weights1 = weights1
                        , biases1 = biases1, weights2 = weights2, biases2 = biases2)
    
    with tf.name_scope('loss'):
        #存储训练轮数的变量
        global_step = tf.Variable(0, trainable = False)
        #add loss to update weights_instance
        
        #计算交叉熵——损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = y, labels = tf.argmax(y_,1)
        )
        #给instance加权
        weighted_cross_entropy = tf.multiply(cross_entropy, w)
        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy)
        #L2正则化损失
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regularization = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy_mean + regularization
        scalar_loss_sum = tf.summary.scalar('loss',loss)
    with tf.name_scope('train'):
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
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 注意这个accuracy是只跟average_y有关的，跟y是无关的
        # 这个运算首先讲一个布尔型的数值转化为实数型，然后计算平均值。这个平均值就是模型在这
        # 一组数据上的正确率 
        #用作validation和test——local
        correct_prediction_float = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction_float)
        scalar_acc_sum = tf.summary.scalar('accuracy',accuracy)
 ##=======================update weights_node_1===================================
    with tf.name_scope('update_transfer_weights'):       
        #本地加权accuracy_weighted_norm
        correct_prediction_float_weighted = tf.multiply(w, correct_prediction_float)
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
        transfer_weight_1 = weights_node_update
        transfer_weight_2 = weights_node_update
        scalar_weight_1_sum = tf.summary.scalar('transfer_weight_1',transfer_weight_1)
        scalar_weight_2_sum = tf.summary.scalar('transfer_weight_2',transfer_weight_2)
    # merged = tf.summary.merge_all()
    # loss_partition = 

with g1.as_default():
 ##=====================Variable Initialization=================================
    #模型输入 shape = [None, Input]其中None表示batch_size大小
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
        w = tf.placeholder(tf.float32, [None, ], name = 'all-instance-weight-local')
    #to calculate the loss of a transferred BATCH, then adjust 'weights_node_1' for network
    with tf.name_scope('transfer_data'):
        x_transfer = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'transfer-x-input')
        y_transfer = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'transfer-y-input')

    #隐藏层参数
    with tf.name_scope('parameters'):
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
    with tf.name_scope('transfer_node_weight'):
        weights_node_init = tf.Variable(
            tf.constant(WEIGHT_INIT, shape = [1], dtype=tf.float32), trainable = True
        )
 ##================================TRAINING=====================================
    #计算前向传播的结果
    with tf.name_scope('inference'):
        y = inference(input_tensor = x, avg_class = None, weights1 = weights1
                        , biases1 = biases1, weights2 = weights2, biases2 = biases2)
    
    with tf.name_scope('loss'):
        #存储训练轮数的变量
        global_step = tf.Variable(0, trainable = False)
        #add loss to update weights_instance
        
        #计算交叉熵——损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = y, labels = tf.argmax(y_,1)
        )
        #给instance加权
        weighted_cross_entropy = tf.multiply(cross_entropy, w)
        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy)
        #L2正则化损失
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regularization = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy_mean + regularization
        scalar_loss_sum = tf.summary.scalar('loss',loss)
    with tf.name_scope('train'):
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
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 注意这个accuracy是只跟average_y有关的，跟y是无关的
        # 这个运算首先讲一个布尔型的数值转化为实数型，然后计算平均值。这个平均值就是模型在这
        # 一组数据上的正确率 
        #用作validation和test——local
        correct_prediction_float = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction_float)
        scalar_acc_sum = tf.summary.scalar('accuracy',accuracy)
 ##=======================update weights_node_1===================================
    with tf.name_scope('update_transfer_weights'):       
        #本地加权accuracy_weighted_norm
        correct_prediction_float_weighted = tf.multiply(w, correct_prediction_float)
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
        transfer_weight_1 = weights_node_update
        transfer_weight_2 = weights_node_update
        scalar_weight_1_sum = tf.summary.scalar('transfer_weight_1',transfer_weight_1)
        scalar_weight_2_sum = tf.summary.scalar('transfer_weight_2',transfer_weight_2)

with tf.Session(graph = g0) as sess:
    # 初始化变量
    writer = tf.summary.FileWriter(log_dir,graph=g0)
    # 准备验证数据。一般在神经网络的训练过程中会通过验证数据要大致判断停止的
    # 条件和评判训练的效果。
    dataset_0, dataset_1, dataset_2, test_data, train_all = load_data()

    #INIT NET_WEIGHT
    with tf.name_scope('init'):
        init_op = tf.initialize_all_variables()
    sess.run(init_op)
    weights_node_1 = sess.run(weights_node_init)
    weights_node_2 = sess.run(weights_node_init)

    for i in range(TRAINING_STEPS):
        #sample and feed training
        feed = dataset_0.sample(n=BATCH_SIZE)
        # feed = train_all.sample(n=BATCH_SIZE)
        local_feed_0 = {
            x: feed.iloc[:,range(INPUT_NODE)].as_matrix(), 
            y_: feed.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
            w: feed.iloc[:, INPUT_NODE+10].as_matrix()
        }

        scalar_loss, _ = sess.run([scalar_loss_sum, train_step], feed_dict=local_feed_0)
        writer.add_summary(scalar_loss,i)

        print(i)



## "NEW TRANSFER" begins, add instance with weights to LOCAL DATA SET, add transferred instance without weights to TRANSFER POOL
            # xe, ye = mnist_datasets.train_even.next_batch(TRANSFER_SIZE)
        if i%(TRANSFER_FRE) == 0 :
            transfer_1_=dataset_1.sample(n=BATCH_SIZE)
            transfer_2_=dataset_2.sample(n=BATCH_SIZE)
            
            #keep weight = 1.0
            # dataset_0 = pd.concat([dataset_0, transfer_1_], axis=0, ignore_index=True)
            # dataset_0 = pd.concat([dataset_0, transfer_2_], axis=0, ignore_index=True)
            
    ##=========================================UPDATE weights_node_i============================================================           
            local_0_ = dataset_0.sample(n=BATCH_SIZE)
            weights_update_feed_1 = {
                x_transfer: transfer_1_.iloc[:,range(INPUT_NODE)].as_matrix(),
                y_transfer: transfer_1_.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(),
                x :local_0_.iloc[:,range(INPUT_NODE)].as_matrix(), 
                y_:local_0_.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
                w: local_0_.iloc[:, INPUT_NODE+10].as_matrix()
            }
            weights_update_feed_2 = {
                x_transfer: transfer_2_.iloc[:,range(INPUT_NODE)].as_matrix(),
                y_transfer: transfer_2_.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(),
                x :local_0_.iloc[:,range(INPUT_NODE)].as_matrix(), 
                y_:local_0_.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
                w: local_0_.iloc[:, INPUT_NODE+10].as_matrix()
            }
            scalar_weight_1, weights_node_1 = sess.run([scalar_weight_1_sum, transfer_weight_1], feed_dict=weights_update_feed_1 )
            scalar_weight_2, weights_node_2 = sess.run([scalar_weight_2_sum, transfer_weight_2], feed_dict=weights_update_feed_2 )
            writer.add_summary(scalar_weight_1,i)
            writer.add_summary(scalar_weight_2,i)
            #更新传送过来的batch中的weights
            transfer_1_.iloc[:, INPUT_NODE+10] = weights_node_1
            transfer_2_.iloc[:, INPUT_NODE+10] = weights_node_2
    ##=========================================RECEIVE & SHUFFLE ALL DATA======================================================
            if weights_node_1 > WEIGHT_THRESHOLD:
                dataset_0 = pd.concat([dataset_0, transfer_1_], axis=0, ignore_index=True)
            if weights_node_2 > WEIGHT_THRESHOLD:
                dataset_0 = pd.concat([dataset_0, transfer_2_], axis=0, ignore_index=True)

        if i % 1 == 0:
            # 计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次
            # 可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更
            # 小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch
            # 会导致计算时间过长甚至发生内存溢出的错误。
            # 注意我们用的是滑动平均之后的模型来跑我们验证集的accuracy
            print("===============TRANSFER WEIGHT: w1 = %f , w2 = %f================ \n" % (weights_node_1, weights_node_2))
            feed = test_data.sample(n=BATCH_SIZE)
            local_feed_0 = {
                x: feed.iloc[:,range(INPUT_NODE)].as_matrix(), 
                y_: feed.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
                w: feed.iloc[:, INPUT_NODE+10].as_matrix()
            }
            scalar_acc, validate_acc = sess.run([scalar_acc_sum, accuracy], feed_dict=local_feed_0)
            writer.add_summary(scalar_acc, i)

            print("After %d training step(s), local accuracy is %g " % (i, validate_acc))
    writer.close()
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
