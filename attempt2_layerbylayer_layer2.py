#import matplotlib as mplib
#import matplotlib.pyplot as plt
import numpy as np
import os
import json
import sys
from six.moves import cPickle as pickle
import tensorflow as tf
assert(tf.VERSION >= '1.0.0')

pickle_file = 'attempt1_layerbylayer_layer1_flipflopcv1_1_outputs.pickle'
trainset_output_file = 'output_trainset.npy'
testset_output_file = 'output_testset.npy'
IMG_DIMS = [50, 50, 64]  #images are 200x200, RGB channel
N_CATS = 6
N_INPUT = 50*50*64

def getParameters(process_no):
    # Network parameters (available at the command line)
    flags = tf.app.flags
    with open('arguments2.json') as data_file:
        data = json.load(data_file)
    flags.DEFINE_integer('instance', 0, "Instance index")
    flags.DEFINE_integer('training_epochs', 40, "Number of training epochs")
    flags.DEFINE_float('learning_rate', 0.0025, "Learning rate")
    flags.DEFINE_integer('train_batch_size', data[process_no]["train_batch_size"], "batch size for learning")
    flags.DEFINE_integer('n_conv_1', data[process_no]["n_conv_1"], "number of filters at first convolutional hidden level")
    flags.DEFINE_integer('n_conv_2', data[process_no]["n_conv_2"], "number of filters at second convolutional hidden level")
    flags.DEFINE_integer('k_size', data[process_no]["k_size"], "Kernel size")
    flags.DEFINE_integer('n_pool_1', data[process_no]["n_pool_1"], "size of pooling nodes at first pooling hidden level")
    flags.DEFINE_integer('n_pool_2', data[process_no]["n_pool_2"], "size of pooling nodes at second pooling hidden level")    
    flags.DEFINE_float('dropout', data[process_no]["dropout"], "Dropout regularization probability")
    flags.DEFINE_string('log_dir', '/tmp/logs', "Directory to write log files in")
    return flags.FLAGS

def buildGraph(flags):
    """
    Bulds the graph
    """
    # tf Graph input (only pictures)
    x = tf.placeholder("float", [None, 50*50*flags.n_conv_1], name="input")
    isTraining = tf.placeholder("bool", [], name="is_training");
    learningrate = tf.placeholder("float", [], name="learningrate")

    # ENCODING LAYER
    inputLayer = tf.reshape(x, [-1, IMG_DIMS[0], IMG_DIMS[1], IMG_DIMS[2]])
    
    # Convolutional Layer #1
    conv2 = tf.layers.conv2d(
            inputs=inputLayer,
            filters=flags.n_conv_2,
            kernel_size=[flags.k_size, flags.k_size],
            padding="SAME",
            activation=tf.nn.relu,
            name='v2')

    # Pooling Layer #1
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=flags.n_pool_2,
            strides=flags.n_pool_2)
    
    # Dropout layer
    dropout = tf.layers.dropout(
            inputs = pool2, 
            rate = flags.dropout, 
            training = isTraining)
    
    #DECODING LAYER
    out = tf.layers.conv2d_transpose(
            inputs = dropout,
            filters = IMG_DIMS[2],
            kernel_size = [flags.k_size, flags.k_size],
            strides = flags.n_pool_2,
            padding = 'SAME',
            name = 'v2_deconv'
        )
    
    # Learning criteria
    cost = tf.reduce_mean(tf.pow(inputLayer - out, 2))
    optimizer = tf.train.RMSPropOptimizer(learningrate).minimize(cost)

    graph = {'input': x, 
             'output': out, 
             'cost': cost, 
             'optimizer': optimizer,
             'learningrate': learningrate
            }
    
    return (graph)


def trainAutoencoder(tfSession, dataset, graph, flags, saver, testset, process_no):
    """
    Trains a built graph. The variables should be previously initialized
    """
    # Define loss and optimizer, minimize the squared error
    x = graph['input']
    cost = graph['cost']
    optimizer = graph['optimizer']
    output = graph['output']
    learningrate = graph['learningrate']
    
    # Training cycle
    isTraining = tfSession.graph.get_tensor_by_name("is_training:0")
    
    savepathname = 'attempt2_layerbylayer_layer2_' + str(process_no) 
    dataset = np.reshape(dataset, [dataset.shape[0], -1])
    nBatches = int(dataset.shape[0]/flags.train_batch_size)
    c = np.inf
    lr = flags.learning_rate
    prev_test_cost = 1.0

    for epoch in range(1, 1+flags.training_epochs):
        # Loop over all batches
        start = 0
        end = flags.train_batch_size
        np.random.shuffle(dataset)
        for i in range(nBatches):
            step = (epoch - 1) * nBatches + i
            # get next batch of images. 
            # since we're minimizing reconstruction errors, labels are ignored
            xBatch = dataset[start:end,:]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = tfSession.run(
                    [optimizer, cost], 
                    feed_dict={x: xBatch, isTraining: True, learningrate: lr})
            savepath = saver.save(tfSession, savepathname)
            start = start + flags.train_batch_size
            end = end + flags.train_batch_size
            print ("Epoch: {0:3d}, step = {1:3d} cost = {2:.6f} variables saved".format(epoch, step, c))    
        # Display logs per epoch step
        print ("Epoch: {0:3d}, cost = {1:.6f}".format(epoch, c))
        testcost = testModel(tfSession, testset, graph, flags)
        if ( testcost > prev_test_cost ):
            lr = float(lr/2)
            print("learningrate decay")
        prev_test_cost = testcost
    return c


def testModel(tfSession, dataset, graph, flags, nImgs=10):
    """
    Tests the models by passing images & displaying reconstructions
    """
    # Applying encoder and decoder over test set
    x = graph['input']
    cost = graph['cost']
    isTraining = tfSession.graph.get_tensor_by_name('is_training:0')
    v2filters = tfSession.graph.get_collection('trainable_variables', 'v2')[0]
    learningrate = graph['learningrate']
    lr = flags.learning_rate

    dataset = np.reshape(dataset, [dataset.shape[0], -1])
    output, v2imgs, cost = tfSession.run(
            [graph['output'], v2filters, cost], 
            feed_dict={x: dataset, isTraining: False, learningrate: lr})
    
    print ("Testingcost = {0:.6f}".format(cost))
    return cost
    # Compare original images with their reconstructions
    #dataset = np.reshape(dataset, [dataset.shape[0],IMG_DIMS[0], IMG_DIMS[1]])
    #output = np.reshape(output, [output.shape[0], IMG_DIMS[0], IMG_DIMS[1]])
    #print (" Test phase - cost = {0:.6f}".format(cost))
    #f, a = mplib.pyplot.subplots(2, nImgs, figsize=(nImgs, 2))
    #for i in range(nImgs):
    #    a[0][i].imshow(dataset[i,:,:])
    #    a[1][i].imshow(output[i,:,:])
    #f.show()
    
    #Figure out how to 'name' the convolutional layers and get their variables
    #v1imgs = v1filters.eval(tfSession)
    #v2imgs = v2filters.eval(tfSession)
    
    #fV1, aV1 = mplib.pyplot.subplots(int(8), int(flags.n_conv_1/8))
   # assert( v1imgs.shape[-1] == flags.n_conv_1 )
    #for i in range(flags.n_conv_1):
     #   aV1[int(i%8)][int(i/8)].imshow(v1imgs[:,:,0,i], cmap='gray')
#        aV1[i%8][i/8].imshow(v1imgs[:,:,0,i], cmap='gray', interpolation='nearest')
    #fV1.show()

    
if __name__ == '__main__':
    
    train_dataset = np.load(trainset_output_file)
    test_dataset = np.load(testset_output_file)
    print('Training set', train_dataset.shape)
    print('Test set', test_dataset.shape)
    
    process_no = int(sys.argv[1])
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

    tc = testModel(sess, test_dataset, graph, flags)
    cost = -1
    for majorEpoch in range(1):
        print ('Major Epoch {0}:'.format(majorEpoch+1))
        #Train model    
        cost = trainAutoencoder(sess, train_dataset, graph, flags, saver, test_dataset, process_no)
        #Test model
        tc = testModel(sess, test_dataset, graph, flags)
    
    print ('Optimization Finished!')
    tc = testModel(sess, test_dataset, graph, flags)
    #testModel(sess, train_dataset, graph, flags)
