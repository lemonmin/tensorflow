import tensorflow as tf

import random
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
numOfRGB = 1
X_img = tf.reshape(X, [-1, 28, 28, numOfRGB])
Y = tf.placeholder(tf.float32, [None, 10])

class buildGraph():
	def __init__(self, filterShapeXY, numOfRGB, numOfFilters, inputDatToConv, strideXY, poolingXY, poolingStrideXY):
		# filter shape set
		self.numOfFilters = numOfFilters
		self.W = tf.Variable(tf.random_normal([filterShapeXY,filterShapeXY,numOfRGB,numOfFilters]))
		self.L = tf.nn.conv2d(inputDatToConv, self.W, strides=[1, strideXY, strideXY, 1], padding='SAME')
		self.L = tf.nn.relu(self.L)
		self.L = tf.nn.max_pool(self.L, ksize=[1, poolingXY, poolingXY, 1], strides=[1, poolingStrideXY, poolingStrideXY, 1], padding='SAME')

class buildFullyConnectedNet():
	def __init__(self, index, inputData, w_shapeX, w_shapeY):
		self.W = tf.get_variable("W"+index, shape=[w_shapeX, w_shapeY], initializer = tf.contrib.layers.xavier_initializer())
		self.b = tf.Variable(tf.random_normal([w_shapeY]))
		self.hypothesis = tf.matmul(inputData, self.W) + self.b

layer1 = buildGraph(3, 1, 32, X_img, 1, 2, 2)
layer2 = buildGraph(3, layer1.numOfFilters, 64, layer1.L, 1, 2, 2)
layer3 = buildGraph(3, layer2.numOfFilters, 64, layer2.L, 1, 2, 2)
print("layer3.L.shape[1] : ",layer3.L.shape[1],"  layer3.L.shape[2] : ",layer3.L.shape[2], "   layer3.L.shape[3] : ",layer3.L.shape[3])

L = tf.reshape(layer3.L, [-1, layer3.L.shape[1]*layer3.L.shape[2]*layer3.L.shape[3]])

print("L shape[1] : ", L.shape[1])
fcLayer1 = buildFullyConnectedNet("1", L, L.shape[1], 100)
fcLayer2 = buildFullyConnectedNet("2", fcLayer1.hypothesis, 100, 100)
fcLayer3 = buildFullyConnectedNet("3", fcLayer2.hypothesis, 100, 10)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fcLayer3.hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
	avg_cost = 0
	total_batch = int(mnist.train.num_examples / batch_size)
	for i in range(total_batch):
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		feed_dict = {X: batch_xs, Y: batch_ys}
		c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
		avg_cost += c / total_batch
	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

correct_prediction = tf.equal(tf.argmax(fcLayer3.hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))