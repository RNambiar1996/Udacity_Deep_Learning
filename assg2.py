from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
	
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']

	del save 

	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
	
	dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set: ', train_dataset.shape, train_labels.shape)
print('Validation set: ', valid_dataset.shape, valid_labels.shape)
print('Test set: ', test_dataset.shape, test_labels.shape)

train_subset = 1000

graph = tf.Graph()

with graph.as_default():
	
	tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
	tf_train_labels = tf.constant(train_labels[:train_subset])
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	weights = tf.Variable(
		tf.truncated_normal([image_size*image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))

	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 801


def accuracy(predictions, labels):
	
	return (100.0*np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
					/predictions.shape[0])

with tf.Session(graph=graph) as session:
	
	tf.global_variables_initializer().run()
	print('Initialised')

	for step in range(num_steps):
	
		_, l, predictions = session.run([optimizer, loss, train_prediction])

		if (step % 100 == 0):

			print('Loss at step %d: %f' %(step, l))
			print('Training accuracy: %.1f%%' %accuracy(
						predictions, train_labels[:train_subset, :]))
			print('Validation accuracy: %.1f%%' %accuracy(
						valid_prediction.eval(), valid_labels))

	print('Test accuracy: %.1f%%' %accuracy(test_prediction.eval(), test_labels))
		

batch_size = 128

graph = tf.Graph()
with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	weights = tf.Variable(
		tf.truncated_normal([image_size*image_size, num_labels]))
	biases = tf.Variable(tf.zeros([num_labels]))
	
	print("Train_dataset: " +str(tf.size(tf_train_dataset)))
	print("Weights_1: " +str(tf.size(weights)))

	logits = tf.matmul(tf_train_dataset, weights) + biases
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
	
num_steps = 3001

with tf.Session(graph=graph) as session:
	
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
		batch_data = train_dataset[offset:(offset+batch_size), :]
		batch_labels = train_labels[offset:(offset+batch_size), :]
		
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict = feed_dict)
		if (step%500 == 0):
			print('Minibatch loss at step %d: %f' %(step, l))
			print('Minibatch accuracy: %.1f%%' %accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' %accuracy(valid_prediction.eval(), valid_labels))
			print('Test accuracy: %.1f%%' %accuracy(test_prediction.eval(), test_labels))

#-----------------------------------------------------------------------------
#1 hidden layer neural network with batch wise training

batch_size = 128
graph = tf.Graph()
with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	num_hidden_nodes = 1024

	#Weights and bias to convert from input layer to first hidden layer
	weights_1 = tf.Variable(
		tf.truncated_normal([image_size*image_size, num_hidden_nodes]))
	biases_1 = tf.Variable(tf.zeros([num_hidden_nodes]))

	#Weights and bias to convert from hidden layer to output layer
	weights_2 = tf.Variable(
		tf.truncated_normal([num_hidden_nodes, num_labels]))
	biases_2 = tf.Variable(tf.zeros([num_labels]))
	
	print("Train_dataset: " +str(tf.size(tf_train_dataset)))
	print("Weights_1: " +str(tf.size(weights_1)))

	a1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
	z1 = tf.nn.relu(a1)

	logits = tf.matmul(z1, weights_2) + biases_2
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	
	a1_v = tf.matmul(tf_valid_dataset, weights_1) + biases_1
	z1_v = tf.nn.relu(a1_v)
	valid_prediction = tf.nn.softmax(
		tf.matmul(z1_v, weights_2) + biases_2)

	a1_t = tf.matmul(tf_test_dataset, weights_1) + biases_1
	z1_t = tf.nn.relu(a1_t)
	test_prediction = tf.nn.softmax(tf.matmul(z1_t, weights_2) + biases_2)
	
num_steps = 3001

with tf.Session(graph=graph) as session:
	
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step*batch_size)%(train_labels.shape[0]-batch_size)
		batch_data = train_dataset[offset:(offset+batch_size), :]
		batch_labels = train_labels[offset:(offset+batch_size), :]
		
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict = feed_dict)
		if (step%500 == 0):
			print('Minibatch loss at step %d: %f' %(step, l))
			print('Minibatch accuracy: %.1f%%' %accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' %accuracy(valid_prediction.eval(), valid_labels))
			print('Test accuracy: %.1f%%' %accuracy(test_prediction.eval(), test_labels))





















	
