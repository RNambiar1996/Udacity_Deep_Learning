# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_subset = 200000
tf_train_dataset_min = train_dataset[:train_subset, :]
tf_train_labels_min =train_labels[:train_subset]

'''
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
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + 0.05*tf.reduce_mean(tf.nn.l2_loss(weights))

	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(
		tf.matmul(tf_valid_dataset, weights) + biases)
	test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 1500

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
'''

#-------------------------------------------------------------------------------
#Neural Network model

batch_size = 50
num_layers = 5
num_hidden_nodes = np.zeros((num_layers+1), dtype = np.int64)
Weights = []
Biases = []

graph = tf.Graph()

with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	keep_prob = tf.Variable(0.5)
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	global_step = tf.Variable(0)

	i =  0
	loss = 0.0

	initializer = tf.contrib.layers.xavier_initializer()
	
	for i in range(num_layers):

		num_hidden_nodes[i] = 1024/(2**i)
		if num_hidden_nodes[i] < num_labels:
			num_hidden_nodes[i] = num_labels
		Biases.append(tf.Variable(initializer([num_hidden_nodes[i]])))

		if i == 0:
			Weights.append(tf.Variable(initializer([image_size*image_size, num_hidden_nodes[i]])))
			
		else:
			Weights.append(tf.Variable(
		initializer([num_hidden_nodes[i-1], num_hidden_nodes[i]])))

		print('w: ' +str(Weights[i]))

	Weights.append(tf.Variable(
	initializer([num_hidden_nodes[i], num_labels])))
	Biases.append(tf.Variable(initializer([num_labels])))	

	print('w: ' +str(Weights[i+1]))
	
	def model(data):
	
		i = 0
		Z = []
	
		Z.append(data)
		for i in range(0,num_layers):
			a = tf.matmul(Z[i], Weights[i]) + Biases[i]
			z = tf.nn.relu(a)
			Z.append(z)
			print('z: ' +str(z))
			
		logits = tf.matmul(Z[num_layers], Weights[num_layers]) + Biases[num_layers]
		print('z: ' +str(logits))
		return logits

	logits = model(tf_train_dataset)
	
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
		
	train_prediction = tf.nn.softmax(logits)

	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))
	
num_steps = 20001

with tf.Session(graph=graph) as session:
	
	tf.global_variables_initializer().run()
	print('Initialized')

	for step in range(num_steps):
		offset = (step*batch_size)%(tf_train_labels_min.shape[0]-batch_size)
		batch_data = tf_train_dataset_min[offset:(offset+batch_size), :]
		batch_labels = tf_train_labels_min[offset:(offset+batch_size), :]
		
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.7}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict = feed_dict)
		if (step%500 == 0):
			print('Minibatch loss at step %d: %f' %(step, l))
			print('Minibatch accuracy: %.1f%%' %accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' %accuracy(valid_prediction.eval(), valid_labels))
			print('Test accuracy: %.1f%%' %accuracy(test_prediction.eval(), test_labels))








