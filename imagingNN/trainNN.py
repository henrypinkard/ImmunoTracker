import tensorflow as tf
import csv, numpy as np, scipy.ndimage as ndi
import os
import shutil
import h5py
import matplotlib.pyplot as plt

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=name)

def bias_variable(shape, value, name):
    return tf.Variable(tf.constant(float(value), shape=[shape]),name=name)


def makeFCLayer(inputlayer, n, bias, name, keep_prob):
    with tf.name_scope(name):
        weight = weight_variable([inputlayer.get_shape()[1].value, n],name+'_weight')
        bias = bias_variable(n, bias, name+'_bias')
        fc = tf.nn.relu(tf.matmul(inputlayer, weight) + bias)
        dropout = tf.nn.dropout(fc, keep_prob)
        return dropout, weight, bias

#load data
filepath = '/Users/henrypinkard/Desktop/2017-1-16_Lymphocyte_iLN_calibration/C_600Rad_70MFP_25_BP_MT_600Rad_30MFP_25BP(MT on this time)_1--Positions as time_333Filtered_e670Candidates.mat'
f = h5py.File(filepath)
designMat = np.zeros(f['nnDesignMatrix'].shape)
f['nnDesignMatrix'].read_direct(designMat)
designMat = designMat.T
excitations = np.zeros(f['excitations'].shape)
f['excitations'].read_direct(excitations)
excitations = excitations.T
f.close()

#filter out abberrant exciations at extremes of distribution
validIndices = np.abs(designMat[:,-1]) < 1.5
excite = excitations[validIndices,0]
inputs = designMat[validIndices]
#shuffle
shuffledIndices = np.random.permutation(np.arange(inputs.shape[0]))
excite = excite[shuffledIndices]
inputs = inputs[shuffledIndices]

inputs = inputs[:,8:]

#split into train/validation
numTrain = int(np.floor(0.8 * inputs.shape[0]))
numVal = inputs.shape[0] - numTrain
trainset = inputs[:numTrain]
valset = inputs[-numVal:]
trainoutputs = excite[:numTrain]
valoutputs = excite[-numVal:]

def readbatch(n, mode = 'train'):
    if mode == 'train':
        set = trainset
        outputs = trainoutputs
    else:
        set = valset
        outputs = valoutputs
    indices = np.random.choice(set.shape[0],size=n)
    return (inputs[indices],np.reshape(outputs[indices],(indices.shape[0],1)))


# Here x and y_ aren't specific values. Rather, they are each a placeholder --
# a value that we'll input when we ask TensorFlow to run a computation.
x = tf.placeholder(tf.float32, shape=[None, inputs.shape[1]])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# We initialized the neuron biases in the second, fourth, and fifth convolutional layers,
# as well as in the fully-connected hidden layers, with the constant 1


keep_prob = tf.placeholder(tf.float32)
fc1, fc1Weight, fc1bias = makeFCLayer(x, 100, 1,'FC1', keep_prob)
fc2Weight = weight_variable([100, 1],'fc2_weight')
fc2bias = bias_variable(1,1,'fc2_bias')
y = tf.matmul(fc1, fc2Weight) + fc2bias


# It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variables are uninitialized
# define loss and accuracy functions
with tf.name_scope('Loss_function'):
    totalSqError = tf.square(y - y_)
    dataloss = tf.reduce_mean(totalSqError)
    tf.summary.scalar('data loss', dataloss)
    # regularization = tf.nn.l2_loss(fc1Weight) + tf.nn.l2_loss(fc2Weight) + tf.nn.l2_loss(fc3Weight)
    # + \   # tf.nn.l2_loss(fc4Weight) + tf.nn.l2_loss(fc5Weight)
    # tf.summary.scalar('weight penalty loss', regularization)
    # weight_decay = 1e-2
    # loss = dataloss + weight_decay*regularization
    loss = dataloss
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
train_step = tf.train.MomentumOptimizer(1e-7, 0.9).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(1e-6).minimize(loss)


with tf.Session() as sess:
    # Operations in TensorFlow don't do anything until you run them. to generate summaries, we need to run
    # all of these summary nodes. Managing them by hand would be tedious, so use tf.summary.merge_all to
    # combine them into a single op that generates all the summary data.
    merged = tf.summary.merge_all()
    #delete old ones
    shutil.rmtree('./log')
    writerTrain = tf.summary.FileWriter('./log/train',graph=sess.graph)
    writerValidation = tf.summary.FileWriter('./log/validation',graph=sess.graph)

    # It is important to realize tf.global_variables_initializer() is a handle to the TensorFlow sub-graph that
    #  initializes all the global variables. Until we call sess.run, the variables are uninitialized.
    # A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    minValidationLoss = 99999999999
    lastSaveIndex = 0

    # train network
    for i in range(100000):
        # print i
        batch = readbatch(1000)
        sess.run(train_step,  feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            #calculate training loss
            alltrain = readbatch(trainset.shape[0],mode='train')
            summary, trainLoss = sess.run([merged,dataloss], feed_dict={x: alltrain[0], y_: alltrain[1], keep_prob: 1.0})
            writerTrain.add_summary(summary, i)
            #calculate validation loss
            allval = readbatch(valset.shape[0], mode='val')
            summary, valLoss = sess.run([merged,dataloss], feed_dict={x: allval[0], y_: allval[1], keep_prob: 1.0})
            writerValidation.add_summary(summary, i)
            line = 'iteration ' + str(i) + '\ttrain loss ' + str(trainLoss) + '\tval loss: ' + str(valLoss)
            print line
            #save model for early stopping
            if(valLoss < minValidationLoss):
                lastSaveIndex = i
                minValidationLoss = valLoss
                saver.save(sess, 'log/model.ckpt')
            if (i - lastSaveIndex > 6000):
                break

    # export weights and biases
    with open('model.csv','wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        variables = [fc1Weight, fc1bias, fc2Weight, fc2bias]
        for var in variables:
            writer.writerow([var.name])
            array = var.eval(sess)
            if (len(array.shape) == 1):
                array = np.reshape(array,(array.shape[0],1))
            for i in np.arange(array.shape[0]):
                writer.writerow(array[i].tolist())
