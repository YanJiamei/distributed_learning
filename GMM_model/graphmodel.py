import tensorflow as tf
import functools
import numpy as np
import pandas as pd
from tensorflow.python.framework import dtypes
INPUT_NODE = 50
OUTPUT_NODE = 10

LAYER1_NODE = 100
BATCH_SIZE = 100
WEIGHT_INIT = 1.0#初始化所有来源样本的权重
WEIGHT_THRESHOLD = 3 #阻止节点之间的数据传输
DISTRIBUTE_NODE_NUM = 3 #参与的节点数目
TRANSFER_FRE = 20 #相互传输的instance的频率，应该也是可以更改的

LEARNING_RATE_BASE = 0.01
TRAINING_STEPS = 3000
THETA = 2.0
logs = './GMM_model/logs/'
train_type = '3000/fre20/noweights/'
# train_type = '1500/fre20/thre1.0/theta2.0/'

log_dir=logs + train_type +'Model/'
log_dir1=logs + train_type +'Model1/'
log_dir2=logs + train_type +'Model2/'
log_tst =logs + train_type +'Test/'
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
        self.accuracy
        self.loss
        self.optimizer
        self.train_step
        self.weighted_cross_entropy
        self.weights_node_update
        self.transfer_weight_1
        self.transfer_weight_2
        self.init_op

        self.flag
        self.scalar_acc
        self.scalar_weight_1
        self.scalar_weight_2
        self.scalar_loss

    #模型输入 shape = [None, Input]其中None表示batch_size大小
    @lazy_property
    def x(self):
        m = tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'x-input')
        return m
    @lazy_property
    def y_(self):
        return tf.placeholder(tf.int64, [None, ], name = 'y-input')
    @lazy_property
    def w(self):    
        return tf.placeholder(tf.float32, [None, ], name = 'all-instance-weight-local')
    #to calculate the loss of a transferred BATCH, then adjust 'weights_node_1' for network
    @lazy_property
    def x_transfer(self):
        return tf.placeholder(tf.float32, [None, INPUT_NODE], name = 'transfer-x-input')
    @lazy_property
    def y_transfer(self):    
        return tf.placeholder(tf.int64, [None, ], name = 'transfer-y-input')

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
        return tf.Variable(dtype= tf.int64,initial_value= 0, trainable=True)
    @lazy_property
    def cross_entropy_mean(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            # logits = tf.clip_by_value(self.y,1e-8,1), labels = self.y_
            logits = self.y, labels = self.y_
        )
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.y, labels = self.y_)
        weighted_cross_entropy = tf.multiply(cross_entropy, self.w)
        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy)   
        return cross_entropy_mean
    @lazy_property
    def loss(self):
        # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        # regularization = regularizer(self.weights1) + regularizer(self.weights2)
        return self.cross_entropy_mean


    # @lazy_property    
    # def learning_rate(self):
    #     learning_rate = tf.train.exponential_decay(
    #         LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行，更新变量时使用的
    #                             # 学习率在这个基础上递减
    #         self.global_step,        # 当前迭代的轮数
    #         # mnist.train.num_examples / BATCH_SIZE, # 过完所有的训练数据需要的迭代次数
    #         1000,
    #         LEARNING_RATE_DECAY # 学习率的衰减速度
    #     )
    #     return learning_rate
    @lazy_property    
    def optimizer(self):
        return tf.train.AdagradOptimizer(learning_rate = LEARNING_RATE_BASE)
    @lazy_property    
    def train_step(self):
        # train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss
        #                 , global_step=self.global_step)
        train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
        return train_step
    @lazy_property
    def correct_prediction_float(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_)
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
        correct_prediction_t = tf.equal(tf.argmax(y_transfer_infer, 1), self.y_transfer)
        correct_prediction_float_t = tf.cast(correct_prediction_t, tf.float32)
        accuracy_transfer_batch = tf.reduce_mean(correct_prediction_float_t)
        return accuracy_transfer_batch
    @lazy_property
    def weighted_cross_entropy(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = self.y, labels = self.y_
        )
        weighted_cross_entropy = tf.multiply(cross_entropy, self.w)
        cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy)
        weighted_cross_entropy = tf.divide(2.0, (1.0 + tf.exp(cross_entropy_mean)))
        return weighted_cross_entropy
    
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
        weights_node_update = THETA*(self.weighted_cross_entropy)*exp_
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
    def flag(self):
        return False
    @lazy_property
    def scalar_acc(self):
        # if self.flag: return tf.summary.scalar('total_acc', self.accuracy)
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

class ArgmaxModel:

    def __init__(self):
        self.graph = tf.Graph()
        self.y_test
        self.y1
        self.y2
        self.y3
        self.argmax
        self.correct_prediction
        self.accuracy
        self.scalar_acc
    @lazy_property
    def y1(self):
        return tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y1-input')

    @lazy_property
    def y2(self):
        return tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y2-input')

    @lazy_property
    def y3(self):
        return tf.placeholder(tf.float32, [None, OUTPUT_NODE], name = 'y3-input')

    @lazy_property
    def y_test(self):
        return tf.placeholder(tf.int64, [None, ], name = 'test-input')
    @lazy_property
    def argmax(self):
        prediction_prob_sum = self.y1 + self.y2 + self.y3
        return tf.arg_max(input=prediction_prob_sum, dimension=1)
    @lazy_property
    def correct_prediction(self):
        equal_num = tf.equal(self.y_test, self.argmax)
        return tf.cast(equal_num, tf.float32)
    @lazy_property
    def accuracy(self):
        return tf.reduce_mean(self.correct_prediction)
    @lazy_property
    def scalar_acc(self):
        return tf.summary.scalar('total_acc', self.accuracy)

def load_datasets():
    dataframe = pd.read_csv('./GMM_model/GMM_data/feature10_100000_10_gmm.csv', index_col=0)
    dataframe = dataframe.sample(frac=1)
    dataframe['weights'] = 1.0
    # print(dataframe)
    train_data = dataframe[0:80000]
    test_data = dataframe[80000:100000]
    train_data_even = train_data.loc[(train_data.gmm_id==2) | (train_data.gmm_id==0) | (train_data.gmm_id==4) | (train_data.gmm_id==6)| (train_data.gmm_id==8),:]
    train_data_odd = train_data.loc[(train_data.gmm_id==1) | (train_data.gmm_id==3) | (train_data.gmm_id==5) | (train_data.gmm_id==0)| (train_data.gmm_id==2),:]
    train_data_mix = train_data.loc[(train_data.gmm_id==1) | (train_data.gmm_id==3) | (train_data.gmm_id==5) | (train_data.gmm_id==7)| (train_data.gmm_id==9),:]
    test_data_odd = test_data.loc[(test_data.gmm_id==1) | (test_data.gmm_id==3) | (test_data.gmm_id==5) | (test_data.gmm_id==7)| (test_data.gmm_id==9),:]
    test_data_even = test_data.loc[(test_data.gmm_id==2) | (test_data.gmm_id==4) | (test_data.gmm_id==6) | (test_data.gmm_id==8)| (test_data.gmm_id==0),:]
    return train_data, test_data, train_data_even,train_data_odd,train_data_mix

train_data, test_data, dataset_0, dataset_1, dataset_2 = load_datasets()
M = Model()
M1 = Model()
M2 = Model()
T = ArgmaxModel()
sess = tf.Session(graph=M.graph)
sess1 = tf.Session(graph=M1.graph)
sess2 = tf.Session(graph=M2.graph)
sess_tst = tf.Session(graph=T.graph)

# log_dir = './GMM_model/GMM_data/logs/'
writer = tf.summary.FileWriter(logdir = log_dir, graph = M.graph)
writer1 = tf.summary.FileWriter(logdir = log_dir1, graph = M1.graph)
writer2 = tf.summary.FileWriter(logdir = log_dir2, graph = M2.graph)
writer_tst = tf.summary.FileWriter(logdir = log_tst, graph = T.graph)
sess.run(M.init_op)
sess1.run(M1.init_op)
sess2.run(M2.init_op)

def transfer(Graph, writer, session, 
                dataset_0,
                dataset_1, dataset_2):
    global_step = session.run(Graph.global_step)
    transfer_1_=dataset_1.sample(n=BATCH_SIZE)
    transfer_2_=dataset_2.sample(n=BATCH_SIZE)
    local_0_ = dataset_0.sample(n=BATCH_SIZE)

    weights_update_feed_1 = {
        Graph.x_transfer: transfer_1_.iloc[:,range(INPUT_NODE)].as_matrix(),
        Graph.y_transfer: transfer_1_.iloc[:,INPUT_NODE].as_matrix(),
        Graph.x :local_0_.iloc[:,range(INPUT_NODE)].as_matrix(), 
        Graph.y_:local_0_.iloc[:,INPUT_NODE].as_matrix(), 
        Graph.w: local_0_.iloc[:, INPUT_NODE+1].as_matrix()
    }
    weights_update_feed_2 = {
        Graph.x_transfer: transfer_2_.iloc[:,range(INPUT_NODE)].as_matrix(),
        Graph.y_transfer: transfer_2_.iloc[:,INPUT_NODE].as_matrix(),
        Graph.x :local_0_.iloc[:,range(INPUT_NODE)].as_matrix(), 
        Graph.y_:local_0_.iloc[:,INPUT_NODE].as_matrix(), 
        Graph.w: local_0_.iloc[:, INPUT_NODE+1].as_matrix()
    }
    #计算差异并更新传输weight
    scalar_weight_1, weights_node_1 = session.run([Graph.scalar_weight_1, Graph.transfer_weight_1], feed_dict=weights_update_feed_1 )
    scalar_weight_2, weights_node_2 = session.run([Graph.scalar_weight_2, Graph.transfer_weight_2], feed_dict=weights_update_feed_2 )
    writer.add_summary(scalar_weight_1,global_step)
    writer.add_summary(scalar_weight_2,global_step)
    #更新传送过来的batch中的weights
    transfer_1_.iloc[:, INPUT_NODE+1] = weights_node_1
    transfer_2_.iloc[:, INPUT_NODE+1] = weights_node_2
    return weights_node_1, weights_node_2, transfer_1_, transfer_2_

def feed(Graph, dataset, batch_size = BATCH_SIZE):
    feed = dataset.sample(n=batch_size)
    # feed = train_all.sample(n=BATCH_SIZE)
    local_feed_0 = {
        Graph.x: feed.iloc[:,range(INPUT_NODE)].as_matrix(), 
        Graph.y_: feed.iloc[:,INPUT_NODE].as_matrix(), 
        Graph.w: feed.iloc[:, INPUT_NODE+1].as_matrix()
    }
    return local_feed_0

count = 0
##train all training in one node
# for i in range(TRAINING_STEPS):

#     local_feed_0 = feed(Graph = M, dataset = train_data)
#     scalar_loss, loss, _ = sess.run([M.scalar_loss, M.loss, M.train_step], feed_dict=local_feed_0)
#     writer.add_summary(scalar_loss,i)

#   ##=====================ACCURACY==========================
#     local_feed_0 = feed(Graph = M, dataset = test_data)
#     M.flag = True
#     scalar_acc, validate_acc = sess.run([M.scalar_acc, M.accuracy], feed_dict=local_feed_0)
#     writer.add_summary(scalar_acc, i)
# writer.close()
# sess.close()

for i in range(TRAINING_STEPS):

    local_feed_0 = feed(Graph = M, dataset = dataset_0)
    scalar_loss, loss, _ = sess.run([M.scalar_loss, M.loss, M.train_step], feed_dict=local_feed_0)
    writer.add_summary(scalar_loss,i)


    local_feed_1 = feed(Graph = M1, dataset = dataset_1)
    scalar_loss1, _ = sess1.run([M1.scalar_loss, M1.train_step], feed_dict=local_feed_1)
    writer1.add_summary(scalar_loss1,i)


    local_feed_2 = feed(Graph = M2, dataset = dataset_2)
    scalar_loss2, _ = sess2.run([M2.scalar_loss, M2.train_step], feed_dict=local_feed_2)
    writer2.add_summary(scalar_loss2,i)
    print(i)
  ##=====================ACCURACY==========================
    local_feed_0 = feed(Graph = M, dataset = test_data)
    scalar_acc, validate_acc = sess.run([M.scalar_acc, M.accuracy], feed_dict=local_feed_0)
    writer.add_summary(scalar_acc, i)

    local_feed_0 = feed(Graph = M1, dataset = test_data)
    scalar_acc, validate_acc = sess1.run([M1.scalar_acc, M1.accuracy], feed_dict=local_feed_0)
    writer1.add_summary(scalar_acc, i)

    local_feed_0 = feed(Graph = M2, dataset = test_data)
    scalar_acc, validate_acc = sess2.run([M2.scalar_acc, M2.accuracy], feed_dict=local_feed_0)
    writer2.add_summary(scalar_acc, i)

  ##=====================SUM ACC=============================
    feed_data = test_data.sample(n = BATCH_SIZE)
    feed_x = feed_data.iloc[:,range(INPUT_NODE)].as_matrix()
    feed_y_ = feed_data.iloc[:,INPUT_NODE].as_matrix()
    y0 = sess.run(M.y, feed_dict={M.x: feed_x})
    y1 = sess1.run(M1.y, feed_dict={M1.x: feed_x})
    y2 = sess2.run(M2.y, feed_dict={M2.x: feed_x})
    feed_test = {
        T.y1 : y0,
        T.y2 : y1,
        T.y3 : y2,
        T.y_test : feed_y_
    }
    scalar_acc_tst, tst_acc = sess_tst.run([T.scalar_acc, T.accuracy], feed_dict = feed_test)
    writer_tst.add_summary(scalar_acc_tst, i)
    print('test acc = ', tst_acc)

  ##=====================TRANSFER & UPDATE==========================
    if i%TRANSFER_FRE == 0 and i>=1 :
#         weights_node_1, weights_node_2, transfer_1_, transfer_2_ = \
#             transfer(Graph = M, writer = writer, session = sess, 
#                                 dataset_0 = dataset_0,
#                                 dataset_1 = dataset_1, dataset_2 = dataset_2)
#         if weights_node_1 > WEIGHT_THRESHOLD:
#             count=count+1
#             dataset_0 = pd.concat([dataset_0, transfer_1_], axis=0, ignore_index=True)
#         if weights_node_2 > WEIGHT_THRESHOLD:
#             dataset_0 = pd.concat([dataset_0, transfer_2_], axis=0, ignore_index=True)
#             count=count+1
#         weights_node_1, weights_node_2, transfer_1_, transfer_2_ = \
#             transfer(Graph = M1, writer = writer1, session = sess1, 
#                                 dataset_0 = dataset_1,
#                                 dataset_1 = dataset_0, dataset_2 = dataset_2)
#         if weights_node_1 > WEIGHT_THRESHOLD:
#             count=count+1
#             dataset_1 = pd.concat([dataset_1, transfer_1_], axis=0, ignore_index=True)
#         if weights_node_2 > WEIGHT_THRESHOLD:
#             count=count+1
#             dataset_1 = pd.concat([dataset_1, transfer_2_], axis=0, ignore_index=True)
#         weights_node_1, weights_node_2, transfer_1_, transfer_2_ = \
#             transfer(Graph = M2, writer = writer2, session = sess2, 
#                                 dataset_0 = dataset_2,
#                                 dataset_1 = dataset_0, dataset_2 = dataset_1)
#         if weights_node_1 > WEIGHT_THRESHOLD:
#             count=count+1
#             dataset_2 = pd.concat([dataset_2, transfer_1_], axis=0, ignore_index=True)
#         if weights_node_2 > WEIGHT_THRESHOLD:
#             count=count+1
#             dataset_2 = pd.concat([dataset_2, transfer_2_], axis=0, ignore_index=True)
# print('transfer times = ', count)

        



      ##================================CONTINUE TRANSFER WEIGHTS = 1 THRE=0==============
        WEIGHT_THRESHOLD = 0
        transfer_0_=dataset_0.sample(n=BATCH_SIZE)
        transfer_1_=dataset_1.sample(n=BATCH_SIZE)
        transfer_2_=dataset_2.sample(n=BATCH_SIZE)
        dataset_0 = pd.concat([dataset_0, transfer_1_], axis=0, ignore_index=True)
        dataset_0 = pd.concat([dataset_0, transfer_2_], axis=0, ignore_index=True)
        dataset_1 = pd.concat([dataset_1, transfer_0_], axis=0, ignore_index=True)
        dataset_1 = pd.concat([dataset_1, transfer_2_], axis=0, ignore_index=True)
        dataset_2 = pd.concat([dataset_2, transfer_0_], axis=0, ignore_index=True)
        dataset_2 = pd.concat([dataset_2, transfer_1_], axis=0, ignore_index=True)



writer.close()
writer1.close()
writer2.close()
writer_tst.close()
sess.close()
sess1.close()
sess2.close()
sess_tst.close()