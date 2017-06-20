import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

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
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ypred, labels=self.y))
        self.obj = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
        self.cor = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ypred, 1))

        tf.global_variables_initializer().run()

    def train(self):
        for _ in range(100):
            x_train, y_train = self.mnist.train.next_batch(100)
            _, l = self.sess.run([self.obj, self.loss], {self.x: x_train, self.y: y_train})
            res = self.cor.eval({self.x: x_train, self.y: y_train})
            print l, sum(res)*1.0/len(res)

    def test(self):
        res = self.cor.eval({self.x: self.mnist.test.images, self.y: self.mnist.test.labels})
        print "Final accuracy: %f"%(sum(res)*1.0/len(res))

class MNISTCNN(MNISTSimple):
    def _get_weight_var(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

    def _get_bias_var(self, shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def _conv(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def _pool(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        x_reshaped = tf.reshape(self.x, [-1,28,28,1])
        w1 = self._get_weight_var([5,5,1,32])
        b1 = self._get_bias_var([32])
        h1 = self._pool(tf.nn.relu(self._conv(x_reshaped, w1) + b1))
        w2 = self._get_weight_var([5,5,32,64])
        b2 = self._get_bias_var([64])
        h2 = self._pool(tf.nn.relu(self._conv(h1, w2) + b2))
        w3 = self._get_weight_var([7*7*64,1024])
        b3 = self._get_bias_var([1024])
        h3 = tf.nn.relu(tf.matmul(tf.reshape(h2, [-1,7*7*64]), w3) + b3)
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        h4 = tf.nn.dropout(h3, self.dropout_keep_prob)
        w5 = self._get_weight_var([1024,10])
        b5 = self._get_bias_var([10])
        self.ypred = tf.matmul(h4, w5) + b5
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.ypred, labels=self.y))
        self.obj = tf.train.AdamOptimizer(1e-4).minimize(loss)

        tf.global_variables_initializer().run()

    def train(self):
        cor = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ypred, 1))
        for i in range(10000):
            x_train, y_train = self.mnist.train.next_batch(100)
            if i % 100 == 0:
                res = cor.eval({self.x: x_train, self.y: y_train, self.dropout_keep_prob: 1.0})
                print "Accuracy at %d: %f"%(i, sum(res)*1.0/len(res))
            self.obj.run({self.x: x_train, self.y: y_train, self.dropout_keep_prob: 0.7})

    def test(self):
        cor = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.ypred, 1))
        res = cor.eval({self.x: self.mnist.test.images, self.y:self.mnist.test.labels, self.dropout_keep_prob: 1.0})
        print "Final accuracy: %f"%(sum(res)*1.0/len(res))

class MNISTCNNLayers(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        self.sess = tf.InteractiveSession()

    def _model_fn(self, features, labels, mode):
        x_reshaped = tf.reshape(features, [-1,28,28,1])
        h1 = tf.layers.conv2d(x_reshaped, 32, [5,5], padding="same", activation=tf.nn.relu)
        h2 = tf.layers.max_pooling2d(h1, 2, 2, padding="same")
        h3 = tf.layers.conv2d(h2, 64, [5,5], padding="same", activation=tf.nn.relu)
        h4 = tf.layers.max_pooling2d(h3, 2, 2, padding="same")
        h5 = tf.layers.dense(tf.reshape(h4, [-1,7*7*64]), 1024, activation=tf.nn.relu)
        h6 = tf.layers.dropout(h5, rate=0.3, training=(mode==learn.ModeKeys.TRAIN))
        ypred = tf.layers.dense(h6, 10)
        predictions = {"classes": tf.argmax(ypred, axis=1),
                       "probabilities": tf.nn.softmax(ypred, dim=1),
                       "logits": ypred}
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ypred, labels=tf.one_hot(labels, depth=10)))
        train_op = None
        if mode == learn.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return model_fn_lib.ModelFnOps(predictions=predictions, loss=loss, train_op=train_op, mode=mode)

    def build_graph(self):
        pass

    def train(self):
        self.mnist_estimator = learn.Estimator(model_fn=self._model_fn, model_dir="/tmp/mnist_convnet_model")
        for i in range(40):
            self.mnist_estimator.fit(x=self.mnist.train.images, y=self.mnist.train.labels, batch_size=100, steps=1)

    def test(self):
        metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}
        print self.mnist_estimator.evaluate(x=self.mnist.test.images, y=self.mnist.test.labels, steps=1, metrics=metrics)

if __name__ == "__main__":
    m = MNISTSimple()
    # m = MNISTCNN()
    # m = MNISTCNNLayers()
    m.build_graph()
    print "\n\nTraining"
    m.train()
    print "\n\nTesting"
    m.test()
