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



def inference(input_tensor, avg_class, weights1, biases1,
                weights2, biases2):
    if (avg_class == None):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return (tf.matmul(layer1, weights2) + biases2)
    else:
        pass

def train(mnist, datanode_0, datanode_1, datanode_2):
##=====================Variable Initialization=================================
    #模型输入 shape = [None, Input]其中None表示batch_size大小
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
    w = tf.placeholder(tf.float32, [None, 1], name = 'all-instance-weight-local')
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
        mnist.train.num_examples / BATCH_SIZE, # 过完所有的训练数据需要的迭代次数
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
    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据要大致判断停止的
        # 条件和评判训练的效果。
        
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
            # x: mnist_datasets.validation_odd.images,
            # y_:  mnist_datasets.validation_odd.labels            
        }
        validate_feed_0 = {
            # x: mnist.validation.images,
            # y_: mnist.validation.labels
            # x: mnist_datasets.validation_odd.images,
            # y_:  mnist_datasets.validation_odd.labels     
            x: datanode_0.validation.images,
            y_: datanode_0.validation.labels       
        }
        validate_feed_1 = {
            # x: mnist_datasets.validation_even.images,
            # y_:  mnist_datasets.validation_even.labels     
            x: datanode_1.validation.images,
            y_: datanode_1.validation.labels        
        }
        validate_feed_2 = {
            # x: mnist_datasets.validation_even.images,
            # y_:  mnist_datasets.validation_even.labels     
            x: datanode_2.validation.images,
            y_: datanode_2.validation.labels        
        }
        # 准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为
        # 模型优劣的最后评价标准。
        #tstfeed改为odd_part
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
            # x: mnist_datasets.test_odd.images,
            # y_:  mnist_datasets.test_odd.labels
        }
        test_feed_0 = {
            # x: mnist_datasets.test_odd.images,
            # y_:  mnist_datasets.test_odd.labels
            x: datanode_0.test.images,
            y_: datanode_0.test.labels 
        }
        test_feed_1 = {
            # x: mnist.test.images,
            # y_: mnist.test.labels
            # x: mnist_datasets.test_even.images,
            # y_:  mnist_datasets.test_even.labels
            x: datanode_1.test.images,
            y_: datanode_1.test.labels 
        }
        test_feed_2 = {
            # x: mnist.test.images,
            # y_: mnist.test.labels
            # x: mnist_datasets.test_even.images,
            # y_:  mnist_datasets.test_even.labels
            x: datanode_2.test.images,
            y_: datanode_2.test.labels 
        }
        # 认真体会这个过程，整个模型的执行流程与逻辑都在这一段
        # 迭代的训练神经网络

        # M：local数据13579先训练一个batch
        # xs, ys = mnist_datasets.train_odd.next_batch(BATCH_SIZE)
        # sess.run(train_step, feed_dict={x: xs, y_: ys})

        #LOAD LOCAL DATA
        # xc = mnist_datasets.train_odd.images
        # yc = mnist_datasets.train_odd.labels
        xc = datanode_0.train.images
        yc = datanode_0.train.labels
        wc = np.ones((yc.shape[0],1), dtype='float64') #跟踪每个instance的权重
        #SHUFFLE
        perm = np.arange(xc.shape[0])
        np.random.shuffle(perm)
        xc = xc[perm]
        yc = yc[perm]
        wc = wc[perm]
        #INIT NET_WEIGHT
        weights_node_1 = sess.run(weights_node_init)
        weights_node_2 = sess.run(weights_node_init)
        #INIT TRANSFER DATA POOL
        xt_pool_1 = np.zeros(shape=(BATCH_SIZE, INPUT_NODE))
        yt_pool_1 = np.zeros(shape=(BATCH_SIZE, OUTPUT_NODE))
        xt_pool_2 = np.zeros(shape=(BATCH_SIZE, INPUT_NODE))
        yt_pool_2 = np.zeros(shape=(BATCH_SIZE, OUTPUT_NODE))
        #TRAINING BEGINS
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1 == 0:
                # 计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次
                # 可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更
                # 小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch
                # 会导致计算时间过长甚至发生内存溢出的错误。
                # 注意我们用的是滑动平均之后的模型来跑我们验证集的accuracy
                print("===============TRANSFER WEIGHT: w1 = %f , w2 = %f================ \n" % (weights_node_1, weights_node_2))

                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy-all is %g " % (i, validate_acc))
                validate_acc_odd = sess.run(accuracy, feed_dict = validate_feed_0)
                print("After %d training step(s), validation accuracy-odd is %g " % (i, validate_acc_odd))
                validate_acc_odd = sess.run(accuracy, feed_dict = validate_feed_1)
                print("After %d training step(s), validation accuracy-even is %g \n" % (i, validate_acc_odd))
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。                 
            local_feed = {
                x: xc[0:BATCH_SIZE], y_: yc[0:BATCH_SIZE], w: wc[0:BATCH_SIZE]
            }
            sess.run(train_step, feed_dict = local_feed)
            
## "NEW TRANSFER" begins, add instance with weights to LOCAL DATA SET, add transferred instance without weights to TRANSFER POOL
            # xe, ye = mnist_datasets.train_even.next_batch(TRANSFER_SIZE)
            xe, ye = datanode_1.train.next_batch(TRANSFER_SIZE)
            we = weights_node_1 * np.ones((TRANSFER_SIZE, 1), dtype='float64')
            pool_i = i*TRANSFER_SIZE%BATCH_SIZE
            xt_pool_1[pool_i:(pool_i+TRANSFER_SIZE)] = xe
            yt_pool_1[pool_i:(pool_i+TRANSFER_SIZE)] = ye

            xo, yo = datanode_2.train.next_batch(TRANSFER_SIZE)
            wo = weights_node_2 * np.ones((TRANSFER_SIZE, 1), dtype='float64')
            pool_i = i*TRANSFER_SIZE%BATCH_SIZE
            xt_pool_2[pool_i:(pool_i+TRANSFER_SIZE)] = xo
            yt_pool_2[pool_i:(pool_i+TRANSFER_SIZE)] = yo
##=========================================UPDATE weights_node_i============================================================           
            #待到传输的instance有一个batch之后 对比accuracy再做调整，这期间传输过来的数据不断地在更新训练，weights-node不变
            if (pool_i == 0) and (i > 0):   
                # xt, yt = mnist_datasets.train_even.next_batch(BATCH_SIZE)
                weights_update_feed_1 = {
                    x_transfer: xt_pool_1,
                    y_transfer: yt_pool_1,
                    x :xc[BATCH_SIZE:2*BATCH_SIZE], 
                    y_: yc[BATCH_SIZE:2*BATCH_SIZE], 
                    w: wc[BATCH_SIZE:2*BATCH_SIZE]
                }
                weights_update_feed_2 = {
                    x_transfer: xt_pool_2,
                    y_transfer: yt_pool_2,
                    x :xc[BATCH_SIZE:2*BATCH_SIZE], 
                    y_: yc[BATCH_SIZE:2*BATCH_SIZE], 
                    w: wc[BATCH_SIZE:2*BATCH_SIZE]
                }
                weights_node_1 = sess.run(weights_node_update, feed_dict=weights_update_feed_1 )
                weights_node_2 = sess.run(weights_node_update, feed_dict=weights_update_feed_2 )
##=========================================RECEIVE & SHUFFLE ALL DATA======================================================
            if weights_node_1 > WEIGHT_THRESHOLD:
                xc = np.vstack((xe,xc))
                yc = np.vstack((ye,yc))
                wc = np.vstack((we,wc))
            if weights_node_2 > WEIGHT_THRESHOLD:
                xc = np.vstack((xo,xc))
                yc = np.vstack((yo,yc))
                wc = np.vstack((wo,wc))

            perm = np.arange(xc.shape[0])
            np.random.shuffle(perm)
            xc = xc[perm]
            yc = yc[perm]
            wc = wc[perm]
##=========================================NEXT TRAINING===================================================================       
        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        # 同样，我们最终的模型用的是滑动平均之后的模型，从这个accuracy函数
        # 的调用就可以看出来了，因为accuracy只与average_y有关
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy-all is %g" % (TRAINING_STEPS, test_acc))
        test_acc_odd = sess.run(accuracy,feed_dict=test_feed_0)
        print("After %d training step(s), test accuracy-odd is %g" % (TRAINING_STEPS, test_acc_odd))
        test_acc_odd = sess.run(accuracy,feed_dict=test_feed_1)
        print("After %d training step(s), test accuracy-even is %g\n=============================================================\n" % (TRAINING_STEPS, test_acc_odd))

# 主程序入口
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    mnist = input_data.read_data_sets("./data", one_hot=True)
    # train(mnist)
    # mnist_13579_train = extract_n_data_sets(mnist.test,label=[1,3,5,7,9])
    # mnist_24680_train = extract_n_data_sets(mnist.test,label=[2,4,6,8,0])
    # mnist_13579_validation = extract_n_data_sets(mnist.validation, label=[1,3,5,7,9])
    # mnist_13579_test = extract_n_data_sets(mnist.validation, label=[1,3,5,7,9])

    # mnist_datasets_load = np.load('./data/mnist_datasets.npy')
    # mnist_datasets = Datasets(train_odd = mnist_datasets_load[0], train_even = mnist_datasets_load[1]
    #                                 , validation_odd = mnist_datasets_load[2], validation_even = mnist_datasets_load[3]
    #                                 , test_odd = mnist_datasets_load[4], test_even = mnist_datasets_load[5])
    
    mnist_datasets_load = np.load('./data/mnist_13579.npy')
    mnist_odd = Datasets(train = mnist_datasets_load[0], validation = mnist_datasets_load[1], test = mnist_datasets_load[2])
    mnist_datasets_load = np.load('./data/mnist_24680.npy')
    mnist_even = Datasets(train = mnist_datasets_load[0], validation = mnist_datasets_load[1], test = mnist_datasets_load[2])
    mnist_datasets_load = np.load('./data/mnist_135702.npy')
    mnist_mix = Datasets(train = mnist_datasets_load[0], validation = mnist_datasets_load[1], test = mnist_datasets_load[2])
    


    train(mnist=mnist, datanode_0 = mnist_odd, datanode_1 = mnist_even, datanode_2 = mnist_mix)
    
# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()
