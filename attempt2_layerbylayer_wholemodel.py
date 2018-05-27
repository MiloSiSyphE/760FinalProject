import tensorflow as tf
assert(tf.VERSION >= '1.0.0')
import numpy as np
#import matplotlib as mplib
#import matplotlib.pyplot as plt
import json
import os
import sys
from six.moves import cPickle as pickle

pickle_file = 'dataset0_trff_teo.pickle'
IMG_DIMS = [100, 100, 1]
N_CATS = 6
N_INPUT = 100*100*1

def getParameters(process_no):
    # Network parameters (available at the command line)
    flags = tf.app.flags
    with open('arguments2.json') as data_file:
        data = json.load(data_file)
    flags.DEFINE_integer('instance', 0, "Instance index")
    flags.DEFINE_integer('training_epochs', 40, "Number of training epochs")
    flags.DEFINE_float('learning_rate', data[process_no]["learning_rate"], "Learning rate")
    flags.DEFINE_integer('train_batch_size', data[process_no]["train_batch_size"], "batch size for learning")
    flags.DEFINE_integer('n_conv_1', data[process_no]["n_conv_1"], "number of filters at first convolutional hidden level")
    flags.DEFINE_integer('k_size_1', data[process_no]["k_size_1"], "Kernel size")
    flags.DEFINE_integer('n_pool_1', data[process_no]["n_pool_1"], "size of pooling nodes at first pooling hidden level")
    flags.DEFINE_integer('n_conv_2', data[process_no]["n_conv_2"], "number of filters at second convolutional hidden level")
    flags.DEFINE_integer('n_pool_2', data[process_no]["n_pool_2"], "size of pooling nodes at second pooling hidden level")
    flags.DEFINE_integer('k_size_2', data[process_no]["k_size_2"], "Kernel size")
    flags.DEFINE_integer('n_pool_3k', data[process_no]["n_pool_3k"], "size of pooling nodes at third pooling hidden level")
    flags.DEFINE_integer('n_pool_3s', data[process_no]["n_pool_3s"], "size of pooling stride at third pooling hidden level")
    flags.DEFINE_integer('n_conv_3', data[process_no]["n_conv_3"], "number of filters at third convolutional hidden level")
    flags.DEFINE_integer('k_size_3', data[process_no]["k_size_3"], "Kernel size")
    flags.DEFINE_integer('n_rec_1', data[process_no]["n_rec_1"], "number of nodes at first hidden layer")
    flags.DEFINE_float('dropout', data[process_no]["dropout"], "Dropout regularization probability")
    flags.DEFINE_integer('n_rec_2', data[process_no]["n_rec_2"], "number of nodes at second hidden layer")
    flags.DEFINE_float('beta1', data[process_no]["beta1"], "Beta1")
    flags.DEFINE_float('beta2', data[process_no]["beta2"], "Beta2")
    flags.DEFINE_float('epsilon', data[process_no]["epsilon"], "epsilon")
    flags.DEFINE_string('log_dir', '/tmp/logs', "Directory to write log files in")
    return flags.FLAGS


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation,:]
  return shuffled_dataset, shuffled_labels


def buildGraph(flags):
    """
    Bulds the graph
    """
    
    x = tf.placeholder("float", [None, N_INPUT], name="input")
    y = tf.placeholder("float", [None, N_CATS], name = "label")
    isTraining = tf.placeholder("bool", [], name="is_training")
    learningrate = tf.placeholder("float", [], name="learningrate")
    
    inputLayer = tf.reshape(x, [-1, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2]])
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
            inputs=inputLayer,
            filters=flags.n_conv_1,
            kernel_size=[flags.k_size_1, flags.k_size_1],
            padding="SAME",
            activation=tf.nn.relu,
            name='v1')

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=flags.n_pool_1,
            strides=flags.n_pool_1)
    
    # Convolutional Layer #1
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=flags.n_conv_2,
            kernel_size=[flags.k_size_2, flags.k_size_2],
            padding="SAME",
            activation=tf.nn.relu,
            name='v2')

    # Pooling Layer #1
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=flags.n_pool_2,
            strides=flags.n_pool_2)
    
    
    pool3 = tf.layers.average_pooling2d(
            inputs=pool2,
            pool_size=flags.n_pool_3k,
            strides=flags.n_pool_3s)
    
    conv3 = tf.layers.conv2d(
            inputs=pool3,
            filters=flags.n_conv_3,
            kernel_size=[flags.k_size_3, flags.k_size_3],
            padding="SAME",
            activation=tf.nn.relu,
            name='cv1')
    
    flat1 = tf.contrib.layers.flatten(conv3)
    
    class1 = tf.layers.dense(
              inputs = flat1,   
              units = flags.n_rec_1,
              activation = tf.nn.relu,
              name='class1'
             )
    
    dropout = tf.layers.dropout(
            inputs = class1, 
            rate = flags.dropout, 
            training = isTraining)
    
    class2 = tf.layers.dense(
              inputs = dropout,   
              units = flags.n_rec_2,
              activation = tf.nn.relu,
              name='class2'
             )
    
    y_pred = tf.layers.dense(
            inputs = class2,
            units = N_CATS,
            name='y_pred'
            )

    # Learning criteria
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningrate, beta1=flags.beta1, beta2=flags.beta2, epsilon=flags.epsilon).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    graph = {'input': x, 
             'output': y_pred, 
             'cost': cost, 
             'optimizer': optimizer,
             'accuracy': accuracy,
             'labels': y,
             'learningrate': learningrate
            }
    
    return (graph)


def trainclassifier(tfSession, dataset, graph, flags, saver, labels, test_dataset, test_labels, process_no):
    """
    Trains a built graph. The variables should be previously initialized
    """
    # Define loss and optimizer, minimize the squared error
    x = graph['input']
    cost = graph['cost']
    optimizer = graph['optimizer']
    accuracy = graph['accuracy']
    y = graph['labels']
    learningrate = graph['learningrate']
    isTraining = tfSession.graph.get_tensor_by_name("is_training:0")
    
    dataset = np.reshape(dataset, [dataset.shape[0], -1])
    labels = (np.arange(1,N_CATS+1) == labels[:,None]).astype(np.float32)
    print('Training set', dataset.shape)
    print('Training labels', labels.shape)     
    savepathname = 'attempt2_layerbylayer_wholemodel_trffteo_' + str(process_no)
    
    # Training cycle
    nBatches = int(dataset.shape[0]/flags.train_batch_size)
    c = np.inf
    lr = flags.learning_rate
    prev_test_cost = 10.0
    
    for epoch in range(1, 1+flags.training_epochs):
        # Loop over all batches
        start = 0
        end = flags.train_batch_size
        dataset, labels = randomize(dataset, labels)
        for i in range(nBatches):
            step = (epoch - 1) * nBatches + i
            # get next batch of images. 
            xBatch = dataset[start:end,:]
            yBatch = labels[start:end, :]
            
            # Run optimization and cost op, get accuracy
            _, c, acc = tfSession.run(
                    [optimizer, cost, accuracy], 
                    feed_dict={x: xBatch, y: yBatch, isTraining: True, learningrate: lr})
            savepath = saver.save(tfSession, savepathname)
            start = start + flags.train_batch_size
            end = end + flags.train_batch_size
            print "Epoch: {0:3d}, step = {1:3d} cost = {2:.6f} accuracy = {3:.6f} variables saved to {4}".format(epoch, step, c, acc, savepath )
        testcost = testModel(tfSession, test_dataset, graph, flags, test_labels)
        if ( epoch % 8 == 0 ):
            lr = float(lr/2)
            print("learningrate decay")
        prev_test_cost = testcost        
    return acc


def testModel(tfSession, dataset, graph, flags, labels):
    """
    Tests the models by checking accuracy on testset
    """
    # Applying encoder and decoder over test set
    x = graph['input']
    acc = graph['accuracy']
    y = graph['labels']  
    isTraining = tfSession.graph.get_tensor_by_name('is_training:0')
    learningrate = graph['learningrate']
    lr = flags.learning_rate
    
    dataset = np.reshape(dataset, [dataset.shape[0], -1])
    labels = (np.arange(1,N_CATS+1) == labels[:,None]).astype(np.float32)
    print('Testing set', dataset.shape)
    print('Testing labels', labels.shape)

    output, accuracy, c = tfSession.run(
            [graph['output'], acc, graph['cost']], 
            feed_dict={x: dataset, y: labels, isTraining: False, learningrate: lr})
    # Display accuracy for each
    print "Testing cost : {0:.6f}, Testing accuracy = {1:.6f}".format(c, accuracy)
    return c
    
    
if __name__ == '__main__':
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_labels = save['train_labels']
        train_dataset = save['train_dataset']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
   
    print('Training set', train_dataset.shape, train_labels.shape)
    #print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    process_no = int(sys.argv[1])
    #process_no = int(1)
    flags = getParameters(process_no)
    #Build graph
    graph = buildGraph(flags)
    #To save and restore variables incase of a crash
    saver = tf.train.Saver(max_to_keep = 1)
    #start an interactive session
    sess = tf.InteractiveSession()

    # Initialize the variables
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, '/Users/raghu/DeepCat/attempt1_layerbylayer_v1')
    
    # Restoring variables from Layer1, Layer2, Classifier models
    tc = testModel(sess, test_dataset, graph, flags, test_labels)
    variables_to_restore = [v for v in tf.trainable_variables() if 'v1' == v.name.split('/')[0]]
    variables_to_restore += [v for v in tf.trainable_variables() if 'v2' == v.name.split('/')[0]]
    saver2 = tf.train.Saver(variables_to_restore)
    saver2.restore(sess, './attempt2_layerbylayer_layer12_0')
    variables_to_restore2 = [v for v in tf.trainable_variables() if 'class' in v.name.split('/')[0]]
    variables_to_restore2 += [v for v in tf.trainable_variables() if 'y_pred' in v.name.split('/')[0]]
    variables_to_restore2 += [v for v in tf.trainable_variables() if 'cv1' in v.name.split('/')[0]]
    saver3 = tf.train.Saver(variables_to_restore2)
    saver3.restore(sess, './attempt2_layerbylayer_classifier5_2')

    testModel(sess, test_dataset, graph, flags, test_labels)
    cost = -1
    for majorEpoch in range(1):
        print 'Major Epoch {0}:'.format(majorEpoch+1)
        #Train model    
        accuracy = trainclassifier(sess, train_dataset, graph, flags, saver, train_labels, test_dataset, test_labels, process_no)
        #Test model
        testModel(sess, test_dataset, graph, flags, test_labels)
    
    print 'Optimization Finished!'
    testModel(sess, test_dataset, graph, flags, test_labels)
