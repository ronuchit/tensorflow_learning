import os
import urllib

import numpy as np
import tensorflow as tf

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"
IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  if not os.path.exists(IRIS_TRAINING):
    raw = urllib.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "w") as f:
      f.write(raw)
  if not os.path.exists(IRIS_TEST):
    raw = urllib.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "w") as f:
      f.write(raw)

  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="/tmp/iris_model")

  def get_data(dataset):
      return tf.constant(dataset.data), tf.constant(dataset.target)

  classifier.fit(input_fn=lambda : get_data(training_set), steps=2000)
  accuracy_score = classifier.evaluate(input_fn=lambda : get_data(test_set), steps=1)["accuracy"]

  print "\nTest Accuracy: {0:f}\n".format(accuracy_score)

if __name__ == "__main__":
    main()
