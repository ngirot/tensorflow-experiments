from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

imageSize = 28*28

K = 200
L = 100
M = 60
N = 30

# Init
print('Initialisation')

x = tf.placeholder(tf.float32, [None, imageSize])

W1 = tf.Variable(tf.truncated_normal([imageSize, K], stddev=0.1)) # weights
b1 = tf.Variable(tf.zeros([K]))            # biais

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
b2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

X = tf.reshape(x, [-1, imageSize])

y1 = tf.nn.relu(tf.matmul(X, W1) + b1)
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)
y = tf.nn.softmax(tf.matmul(y4, W5) + b5)

# Training
print('Training')
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluation
print('Evaluation')
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
