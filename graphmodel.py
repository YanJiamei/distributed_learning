import tensorflow as tf
import functools
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
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with self.graph.as_default():
                with tf.variable_scope(function.__name__):
                    setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Model:

    def __init__(self):
        self.graph = tf.Graph()
        self.x
        self.y_
        self.w
        self.x_transfer
        self.y_transfer
        self.weights1
        self.weights2
        self.biases1
        self.biases2
        self.global_step
        self.init_op
        self.train_step
        self.accuracy
        self.loss
        self.weights_node_update

        self.scalar_acc
        self.scalar_weight_1
        self.scalar_weight_2
        self.scalar_loss

    #模型输入 shape = [None, Input]其中None表示batch_size大小
    @lazy_property
    def x(self):
        return tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
    @lazy_property
    def y_(self):
        return tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y-input')
    @lazy_property
    def w(self):    
        return tf.placeholder(tf.float32, [None, ], name = 'all-instance-weight-local')
    #to calculate the loss of a transferred BATCH, then adjust 'weights_node_1' for network
    @lazy_property
    def x_transfer(self):
        return tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'transfer-x-input')
    @lazy_property
    def y_transfer(self):    
        return tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'transfer-y-input')

    #隐藏层参数    @lazy_property
    @lazy_property
    def weights1(self):
        return tf.Variable(
            tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev = 0.1) #初始化
        )
    @lazy_property
    def biases1(self):
        return tf.Variable(
            tf.constant(0.1, shape = [LAYER1_NODE])
        )
    @lazy_property
    def weights2(self):
        return tf.Variable(
            tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1)
        )
    @lazy_property
    def biases2(self):
        return tf.Variable(
            tf.constant(0.1, shape = [OUTPUT_NODE])
        )

    def inference(self, input_tensor, weights1, biases1,
                weights2, biases2):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return (tf.matmul(layer1, weights2) + biases2)

    @lazy_property
    def y(self):
        return self.inference(input_tensor = self.x, weights1 = self.weights1
                        , biases1 = self.biases1, weights2 = self.weights2, biases2 = self.biases2)

    @lazy_property
    def global_step(self):
        return tf.Variable(0, trainable=False)
    @lazy_property
    def cross_entropy_mean(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.y, labels = tf.argmax(self.y_,1)
        )
        weighted_cross_entropy = tf.multiply(cross_entropy, self.w)
        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy)   
        return cross_entropy_mean
    @lazy_property
    def loss(self):
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        regularization = regularizer(self.weights1) + regularizer(self.weights2)
        return self.cross_entropy_mean +  regularization


    @lazy_property    
    def learning_rate(self):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行，更新变量时使用的
                                # 学习率在这个基础上递减
            self.global_step,        # 当前迭代的轮数
            # mnist.train.num_examples / BATCH_SIZE, # 过完所有的训练数据需要的迭代次数
            1000,
            LEARNING_RATE_DECAY # 学习率的衰减速度
        )
        return learning_rate
    @lazy_property    
    def train_step(self):
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss
                        , global_step=self.global_step
                        , var_list=[self.weights1, self.biases1, self.weights2, self.biases2])
        return train_step
    @lazy_property
    def correct_prediction_float(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        # 注意这个accuracy是只跟average_y有关的，跟y是无关的
        # 这个运算首先讲一个布尔型的数值转化为实数型，然后计算平均值。这个平均值就是模型在这
        # 一组数据上的正确率 
        #用作validation和test——local
        correct_prediction_float = tf.cast(correct_prediction, tf.float32)
        return correct_prediction_float
        
    @lazy_property        
    def accuracy(self):
        return tf.reduce_mean(self.correct_prediction_float)
    @lazy_property
    def accuracy_transfer_batch(self):
        #传输进来的一个batch的accuracy_transfer_batch
        y_transfer_infer = self.inference(input_tensor = self.x_transfer, weights1 = self.weights1
                        , biases1 = self.biases1, weights2 = self.weights2, biases2 = self.biases2)
        correct_prediction_t = tf.equal(tf.argmax(y_transfer_infer, 1), tf.argmax(self.y_transfer, 1))
        correct_prediction_float_t = tf.cast(correct_prediction_t, tf.float32)
        accuracy_transfer_batch = tf.reduce_mean(correct_prediction_float_t)
        return accuracy_transfer_batch
    @lazy_property
    def accuracy_weighted_norm(self):
        #本地加权的acc，查看数据差异
        correct_prediction_float_weighted = tf.multiply(self.w, self.correct_prediction_float)
        accuracy_weighted = tf.reduce_sum(correct_prediction_float_weighted)
        accuracy_weighted_norm = tf.divide(accuracy_weighted, tf.reduce_sum(self.w))
        return accuracy_weighted_norm
    @lazy_property    
    def weights_node_update(self):
        exp_ = tf.exp(1.0-self.accuracy_transfer_batch)
        weights_node_update = (self.accuracy_weighted_norm)*exp_
        return weights_node_update
    @lazy_property    
    def transfer_weight_1(self):
        return self.weights_node_update
    @lazy_property
    def transfer_weight_2(self):
        return self.weights_node_update
    @lazy_property
    def scalar_loss(self):
        return tf.summary.scalar('loss', self.loss)
    @lazy_property
    def scalar_acc(self):
        return tf.summary.scalar('accuracy', self.accuracy)
    @lazy_property
    def scalar_weight_1(self):
        return tf.summary.scalar('transfer_weight_1', self.transfer_weight_1)
    @lazy_property
    def scalar_weight_2(self):
        return tf.summary.scalar('transfer_weight_2', self.transfer_weight_2)

    @lazy_property
    def init_op(self):
        return tf.initialize_all_variables()
    # @lazy_property
    # def prediction(self):
    #     data_size = int(self.data.get_shape()[1])
    #     target_size = int(self.target.get_shape()[1])
    #     weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
    #     bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
    #     incoming = tf.matmul(self.data, weight) + bias
    #     return tf.nn.softmax(incoming)

    # @lazy_property
    # def optimize(self):
    #     cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
    #     optimizer = tf.train.RMSPropOptimizer(0.03)
    #     return optimizer.minimize(cross_entropy)

    # @lazy_property
    # def error(self):
    #     mistakes = tf.not_equal(
    #         tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
    #     return tf.reduce_mean(tf.cast(mistakes, tf.float32))
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets import mnist as MDataset
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes
import collections
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




M = Model()
M1 = Model()
M2 = Model()
sess = tf.Session(graph=M.graph)
sess1 = tf.Session(graph=M1.graph)
sess2 = tf.Session(graph=M2.graph)
log_dir='./logs/Model/'
log_dir1='./logs/Model1/'
log_dir2='./logs/Model2/'

writer = tf.summary.FileWriter(logdir = log_dir, graph = M.graph)
writer1 = tf.summary.FileWriter(logdir = log_dir1, graph = M1.graph)
writer2 = tf.summary.FileWriter(logdir = log_dir2, graph = M2.graph)
sess.run(M.init_op)
sess1.run(M1.init_op)
sess2.run(M2.init_op)
dataset_0, dataset_1, dataset_2, test_data, train_all = load_data()
def transfer(dataset_0 , dataset_1 , dataset_2 
            , transfer_1_ , transfer_2_ ):

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
    return scalar_weight_1, weights_node_1, scalar_weight_2, weights_node_2
for i in range(TRAINING_STEPS):
    #sample and feed training
    feed = dataset_0.sample(n=BATCH_SIZE)
    # feed = train_all.sample(n=BATCH_SIZE)
    local_feed_0 = {
        M.x: feed.iloc[:,range(INPUT_NODE)].as_matrix(), 
        M.y_: feed.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
        M.w: feed.iloc[:, INPUT_NODE+10].as_matrix()
    }

    scalar_loss, _ = sess.run([M.scalar_loss, M.train_step], feed_dict=local_feed_0)
    writer.add_summary(scalar_loss,i)

    feed = dataset_1.sample(n=BATCH_SIZE)
    # feed = train_all.sample(n=BATCH_SIZE)
    local_feed_1 = {
        M1.x: feed.iloc[:,range(INPUT_NODE)].as_matrix(), 
        M1.y_: feed.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
        M1.w: feed.iloc[:, INPUT_NODE+10].as_matrix()
    }

    scalar_loss1, _ = sess.run([M1.scalar_loss, M1.train_step], feed_dict=local_feed_1)
    writer1.add_summary(scalar_loss1,i)

    feed = dataset_2.sample(n=BATCH_SIZE)
    # feed = train_all.sample(n=BATCH_SIZE)
    local_feed_2 = {
        M2.x: feed.iloc[:,range(INPUT_NODE)].as_matrix(), 
        M2.y_: feed.iloc[:,range(INPUT_NODE, INPUT_NODE+10)].as_matrix(), 
        M2.w: feed.iloc[:, INPUT_NODE+10].as_matrix()
    }

    scalar_loss2, _ = sess.run([M2.scalar_loss, M2.train_step], feed_dict=local_feed_2)
    writer2.add_summary(scalar_loss2,i)
    print(i)
    if i%TRANSFER_FRE == 0:
        scalar_weight_1, weights_node_1, scalar_weight_2, weights_node_2 = transfer(dataset_0 = dataset_0, dataset_1 = dataset_1, dataset_2 = dataset_2
            , transfer_1_ = transfer_1_, transfer_2_ = transfer_2_)
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
        
writer.close()