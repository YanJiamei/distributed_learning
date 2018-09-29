import tensorflow as tf

# g1 = tf.Graph()
# g2 = tf.Graph()

# with g1.as_default():
#     a = tf.constant([1,2,3])
#     b = tf.constant([4,5,6])
#     c1 = tf.add(a,b)

# with g2.as_default():
#     a = tf.constant([11,2,31])
#     b = tf.constant([41,51,6])
#     c = tf.add(a,b)

# with tf.Session(graph=g1) as sess1:
#     print(sess1.run(c1))
#     print(sess1.run(a))
import functools

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

    def __init__(self, data, target):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data = data
            self.target = target
            self.prediction
            self.a
            self.b
            
    @lazy_property
    def a(self):
        return tf.placeholder(tf.int32, [None,], name = 'a')

    @lazy_property
    def b(self):
        with tf.name_scope('nnn'):
            return tf.constant(self.target)

    @lazy_property
    def prediction(self):
        return tf.add(self.a,self.b)
    # @property
    # def graph(self):
    #     return self.graph

M1 = Model([1,2,3],[4,5,6])
M2 = Model([1,1],[2,2])
sess1 = tf.Session(graph=M1.graph)
sess2 = tf.Session(graph=M2.graph)
print(sess1.run(M1.prediction, feed_dict={M1.a: [9,9,9]}))
print(M2.b)