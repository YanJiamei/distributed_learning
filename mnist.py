import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist 
import numpy as np
from tensorflow.python.framework import dtypes
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1,
                weights2, biases2):
    if (avg_class == None):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return (tf.matmul(layer1, weights2) + biases2)
    else:
        pass

def train(mnist):
    #模型输入 shape = [None, Input]其中None表示batch_size大小
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
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
    #add weights_instance to train and boost
    weights_instance = tf.Variable(
        tf.constant(1/BATCH_SIZE, shape = [OUTPUT_NODE])
    )
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
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
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
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 注意这个accuracy是只跟average_y有关的，跟y是无关的
    # 这个运算首先讲一个布尔型的数值转化为实数型，然后计算平均值。这个平均值就是模型在这
    # 一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据要大致判断停止的
        # 条件和评判训练的效果。
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        # 准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为
        # 模型优劣的最后评价标准。
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }
        # 认真体会这个过程，整个模型的执行流程与逻辑都在这一段
        # 迭代的训练神经网络
        for i in range(TRAINING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次
                # 可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更
                # 小的batch。当神经网络模型比较复杂或者验证数据比较大时，太大的batch
                # 会导致计算时间过长甚至发生内存溢出的错误。
                # 注意我们用的是滑动平均之后的模型来跑我们验证集的accuracy
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy is %g " % (i, validate_acc))

            # 产生这一轮使用的一个batch的训练数据，并运行训练过程。
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率。
        # 同样，我们最终的模型用的是滑动平均之后的模型，从这个accuracy函数
        # 的调用就可以看出来了，因为accuracy只与average_y有关
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy is %g" % (TRAINING_STEPS, test_acc))

def extract_n_data_sets(datasets, label = 0):
    test_data_set_image = datasets.test.images
    test_data_set_label = datasets.test.labels
    extract_images = np.array([])
    extract_labels = np.array([])
    cnt = 0
    for i in range(10000):
        if np.argmax (test_data_set_label[i]) == label:
            cnt += 1
            extract_images = np.append(extract_images , test_data_set_image[i])
            extract_labels = np.append(extract_labels , test_data_set_label[i])
    extract_images = extract_images.astype(np.float32)
        # return mnist.DataSet(extract_images, extract_labels, dtype = dtypes.float32, reshape = True)
    return extract_images.reshape(cnt,784), extract_labels.reshape(cnt,10)
# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据。
    mnist = input_data.read_data_sets("./data", one_hot=True)
    x0,y0 = extract_n_data_sets(mnist,label=0)
    x1,y1 = extract_n_data_sets(mnist,label=1)
    print ('ok')
# TensorFlow提供的一个主程序入口，tf.app.run会调用上面定义的main函数
if __name__ == "__main__":
    tf.app.run()