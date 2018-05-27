import tensorflow as tf
assert(tf.VERSION >= '1.0.0')
import numpy as np
#import matplotlib as mplib
#import matplotlib.pyplot as plt
import json
import os
import sys
from six.moves import cPickle as pickle

pickle_file = 'dataset0.pickle'
output_testfile = 'output_layer12_testset.npy'
output_trainfile = 'output_layer12_trainset.npy'
IMG_DIMS = [25, 25, 128]
N_CATS = 6
N_INPUT = 25*25*128

def getParameters(process_no):
    # Network parameters (available at the command line)
    flags = tf.app.flags
    with open('arguments.json') as data_file:
        data = json.load(data_file)
    flags.DEFINE_integer('instance', 0, "Instance index")
    flags.DEFINE_integer('training_epochs', 40, "Number of training epochs")
    flags.DEFINE_float('learning_rate', 0.0025, "Learning rate")
    flags.DEFINE_integer('train_batch_size', data[process_no]["train_batch_size"], "batch size for learning")
    flags.DEFINE_integer('n_conv_1', data[process_no]["n_conv_1"], "number of filters at first convolutional hidden level")
    flags.DEFINE_integer('n_conv_2', data[process_no]["n_conv_2"], "number of filters at second convolutional hidden level")
    flags.DEFINE_integer('n_rec_1', data[process_no]["n_rec_1"], "number of nodes at first hidden layer")
    flags.DEFINE_integer('n_rec_2', data[process_no]["n_rec_2"], "number of nodes at second hidden layer")
    flags.DEFINE_integer('k_size', data[process_no]["k_size"], "Kernel size")
    flags.DEFINE_integer('n_pool_1', data[process_no]["n_pool_1"], "size of pooling nodes at first pooling hidden level")
    flags.DEFINE_integer('n_pool_2', data[process_no]["n_pool_2"], "size of pooling nodes at second pooling hidden level")
    flags.DEFINE_float('dropout', data[process_no]["dropout"], "Dropout regularization probability")
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
    
    class1 = tf.layers.dense(
              inputs = x,   
              units = flags.n_rec_1,
              activation = tf.nn.relu
             )
    
    class2 = tf.layers.dense(
              inputs = class1,   
              units = flags.n_rec_2,
              activation = tf.nn.relu
             )
    
    y_pred = tf.layers.dense(
            inputs = class2,
            units = N_CATS
            )

    # Learning criteria
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learningrate).minimize(cost)
    
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
    savepathname = 'attempt2_layerbylayer_classifier_2hiddenlayer_' + str(process_no)
    
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
        if ( testcost > prev_test_cost ):
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
        #valid_dataset = save['valid_dataset']
        #valid_labels = save['valid_labels']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

    train_dataset = np.load(output_trainfile)
    test_dataset = np.load(output_testfile)    
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
