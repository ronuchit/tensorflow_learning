from skdata.mnist.views import OfficialVectorClassification
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from IPython import embed as shell



# data = OfficialVectorClassification()
# trIdx = data.sel_idxs[:]
# np.random.shuffle(trIdx)
# writer = tf.python_io.TFRecordWriter("mnist.tfrecords")
# for example_idx in tqdm(trIdx):
#     features = data.all_vectors[example_idx]
#     label = data.all_labels[example_idx]
#     example = tf.train.Example(
#         features=tf.train.Features(
#           feature={
#             "label_test": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
#             "image_test": tf.train.Feature(int64_list=tf.train.Int64List(value=features.astype("int64")))}))
#     serialized = example.SerializeToString()
#     writer.write(serialized)


# filename = "mnist.tfrecords"
# filename_test = "mnist.tfrecords.test"
# num_test = 10000
# sess = tf.InteractiveSession()
# reader = tf.TFRecordReader()
# reader_test = tf.TFRecordReader()
# queue = tf.train.string_input_producer([filename])
# queue_test = tf.train.string_input_producer([filename_test])
# _, v = reader.read(queue)
# _, v_test = reader_test.read(queue_test)
# example = tf.parse_single_example(v, {"label": tf.FixedLenFeature([1], tf.int64), "image": tf.FixedLenFeature([784], tf.int64)})
# example_test = tf.parse_single_example(v_test, {"label_test": tf.FixedLenFeature([1], tf.int64), "image_test": tf.FixedLenFeature([784], tf.int64)})
# image, label = example["image"], example["label"]
# image_test, label_test = example_test["image_test"], example_test["label_test"]
# label, label_test = label[0], label_test[0]

# images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=100, capacity=2000, min_after_dequeue=1000)
# images_test, labels_test = tf.train.batch([image_test, label_test], batch_size=num_test, capacity=num_test)

# w = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# ypred = tf.matmul(tf.cast(images_batch, tf.float32)/255, w) + b
# ypred_test = tf.matmul(tf.cast(images_test, tf.float32)/255, w) + b
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ypred, labels=tf.one_hot(labels_batch, depth=10)))
# obj = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# cor = tf.equal(tf.argmax(ypred, 1), labels_batch)
# cor_test = tf.equal(tf.argmax(ypred_test, 1), labels_test)

# tf.local_variables_initializer().run()
# tf.global_variables_initializer().run()
# tf.train.start_queue_runners()

# for _ in range(100):
#     l, _ = sess.run([loss, obj])
#     res = cor.eval()
#     print l, sum(res)*1.0/len(res), len(res)
# print
# res = cor_test.eval()
# print sum(res)*1.0/len(res), len(res)



filename = "mnist.tfrecords"
filename_test = "mnist.tfrecords.test"
num_test = 10000
sess = tf.InteractiveSession()
reader = tf.TFRecordReader()
reader_test = tf.TFRecordReader()
queue = tf.train.string_input_producer([filename])
queue_test = tf.train.string_input_producer([filename_test])
_, v = reader.read(queue)
_, v_test = reader_test.read(queue_test)
v_test_batched = tf.train.batch([v_test], batch_size=num_test, capacity=num_test)

example = tf.parse_single_example(v, {"label": tf.FixedLenFeature([1], tf.int64), "image": tf.FixedLenFeature([784], tf.int64)})
example_test_batched = tf.parse_example(v_test_batched, {"label_test": tf.FixedLenFeature([1], tf.int64), "image_test": tf.FixedLenFeature([784], tf.int64)})
image, label = example["image"], example["label"]
image_test, label_test = example_test_batched["image_test"], example_test_batched["label_test"]
label = label[0]

images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=100, capacity=2000, min_after_dequeue=1000)

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
ypred = tf.matmul(tf.cast(images_batch, tf.float32)/255, w) + b
ypred_test = tf.matmul(tf.cast(image_test, tf.float32)/255, w) + b
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ypred, labels=tf.one_hot(labels_batch, depth=10)))
obj = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
cor = tf.equal(tf.argmax(ypred, 1), labels_batch)
cor_test = tf.equal(tf.argmax(ypred_test, 1), tf.squeeze(label_test))

tf.local_variables_initializer().run()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for _ in range(100):
    l, _ = sess.run([loss, obj])
    res = cor.eval()
    print l, sum(res)*1.0/len(res), len(res)
print
res = cor_test.eval()
print sum(res)*1.0/len(res), len(res)
