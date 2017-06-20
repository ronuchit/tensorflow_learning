import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MNISTSimple(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.sess = tf.InteractiveSession()

    def build_graph(self):
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.ypred = tf.matmul(self.x,w) + b
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ypred, labels=self.y))
        self.obj = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        tf.global_variables_initializer().run()

    def train(self):
        for _ in range(1000):
            x_train, y_train = self.mnist.train.next_batch(100)
            self.obj.run({self.x: x_train, self.y: y_train})

    def test(self):
        cor = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ypred, 1))
        res = cor.eval({self.x: self.mnist.test.images, self.y:self.mnist.test.labels})
        print "Final accuracy: %f"%(sum(res)*1.0/len(res))

class MNISTCNN(MNISTSimple):
    def _get_weight_var(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

    def _get_bias_var(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def _conv(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def _pool(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="data")
        self.y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
        x_reshaped = tf.reshape(self.x, [-1,28,28,1])
        with tf.name_scope("conv1"):
            w1 = self._get_weight_var([5,5,1,32], "W1")
            b1 = self._get_bias_var([32], "b1")
            h1 = self._pool(tf.nn.relu(self._conv(x_reshaped, w1) + b1))

        with tf.name_scope("conv2"):
            w2 = self._get_weight_var([5,5,32,64], "W2")
            b2 = self._get_bias_var([64], "b2")
            h2 = self._pool(tf.nn.relu(self._conv(h1, w2) + b2))

        with tf.name_scope("fc1"):
            w3 = self._get_weight_var([7*7*64,1024], "W3")
            b3 = self._get_bias_var([1024], "b3")
            h3 = tf.nn.relu(tf.matmul(tf.reshape(h2, [-1,7*7*64]), w3) + b3)
            self.dropout_keep_prob = tf.placeholder(tf.float32)
            h4 = tf.nn.dropout(h3, self.dropout_keep_prob)

        with tf.name_scope("fc2"):
            w5 = self._get_weight_var([1024,10], "W5")
            b5 = self._get_bias_var([10], "b5")
            self.ypred = tf.matmul(h4, w5) + b5

        with tf.name_scope("objective"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ypred, labels=self.y))
            tf.summary.scalar("loss", loss)
            self.obj = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
        tf.global_variables_initializer().run()

    def train(self):
        cor = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ypred, 1))
        for i in range(10000):
            x_train, y_train = self.mnist.train.next_batch(100)
            if i % 100 == 0:
                res = cor.eval({self.x: x_train, self.y: y_train, self.dropout_keep_prob: 1.0})
                print "Accuracy at %d: %f"%(i, sum(res)*1.0/len(res))
            # self.obj.run({self.x: x_train, self.y: y_train, self.dropout_keep_prob: 0.7})
            _, summary = self.sess.run([self.obj, self.merged], {self.x: x_train, self.y: y_train, self.dropout_keep_prob: 0.7})
            self.writer.add_summary(summary, global_step=i)

    def test(self):
        cor = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ypred, 1))
        res = cor.eval({self.x: self.mnist.test.images, self.y:self.mnist.test.labels, self.dropout_keep_prob: 1.0})
        print "Final accuracy: %f"%(sum(res)*1.0/len(res))

if __name__ == "__main__":
    # m = MNISTSimple()
    m = MNISTCNN()
    m.build_graph()
    m.train()
    m.test()
