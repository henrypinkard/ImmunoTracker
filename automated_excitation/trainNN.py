import tensorflow as tf
import csv, numpy as np, scipy.ndimage as ndi
import os
import shutil
import h5py
import matplotlib.pyplot as plt

#load data
def readIntoArray(structgroup, name):
    arr = np.zeros(structgroup[name].shape)
    structgroup[name].read_direct(arr)
    return arr.T

filepath = '/Users/henrypinkard/Desktop/2017-4-4/MT_55_600_30_2--Positions as time_e670Candidates.mat'
# filepath = '/Users/henrypinkard/Desktop/2017-4-4/C_40_600_70_1--Positions as time_GFPCandidates.mat'
# filepath = '/Users/henrypinkard/Desktop/2017-4-4/OldTest.mat'

f = h5py.File(filepath)
struct = f.get('excitationNNData')
#read variables
normalizedTilePosition = readIntoArray(struct, 'normalizedTilePosition')
normalizedBrightness = readIntoArray(struct, 'normalizedBrightness')
# normalizedBrightnessMedian = readIntoArray(struct, 'normalizedBrightnessMedian')
# timeIndex = readIntoArray(struct, 'timeIndices')
distancesToInterpolation = readIntoArray(struct, 'distancesToInterpolation')
# distancesToInterpolationSP = readIntoArray(struct, 'distancesToInterpolarionSP') #centered at local stage position
excitations = readIntoArray(struct, 'excitations')
f.close()


#make design matrix for training
#use single measure to calibrate vignetting
# distanceToFOVCenter = np.sqrt(np.sum(np.square(normalizedTilePosition - 0.5),axis=1))
# distanceToFOVCenterNormalized = np.reshape((distanceToFOVCenter - np.mean(distanceToFOVCenter)) / np.std(distanceToFOVCenter),(-1,1))
#bin distances into histrograms
#(cell, theta, phi) -- distributions hsould be invariant to theta but not phi
def binsurfacedistance(dist):
     #first one is vertical distance--all the same
     histdesignmat = dist[:,0,0]
     histdesignmat = np.reshape(histdesignmat,(histdesignmat.shape[0],1))
     #bin remaining phi angles into histogram
     for i in [2]:
        # binsize = 15
        binmax = 350
        numbins = 12
        # binmax = np.percentile(np.ravel(dist[:,:,i]),95)
        # binedges = np.linspace(0, binmax, binmax/binsize + 1)
        binedges = np.power(np.linspace(0, 1, numbins+1), 1.5) * binmax
        distancesforcurrentphi = dist[:, :, i]
        counts = np.apply_along_axis(lambda x: np.histogram(x, binedges)[0], 1, distancesforcurrentphi)
        histdesignmat = np.concatenate((histdesignmat, counts), axis=1)
        histdesignmat = counts #overwrite phi0

     return histdesignmat
     # return np.reshape(histdesignmat[:,0],(histdesignmat.shape[0],1))
     # return histdesignmat[:,[0,1,5]]

distancefeatures = binsurfacedistance(distancesToInterpolation)
# distancefeatures = binsurfacedistance(distancesToInterpolationSP) #use distances centered at stage position
#save distance normalization params
distancefeaturesmean = np.mean(distancefeatures,axis=0)
distancefeaturessd = np.std(distancefeatures,axis=0)
distancefeaturesnormalized = (distancefeatures - distancefeaturesmean) / distancefeaturessd
# designmatrix = np.concatenate((distancefeaturesnormalized, normalizedBrightness),axis=1 )
designmatrix = np.concatenate((distancefeaturesnormalized, normalizedTilePosition,  normalizedBrightness),axis=1 )



# excitations = np.random.randint(0,255,(excitations.shape[0],1)) #replace with random values
# excitations = excitations[:,0] # take maitai
#remove saturated excitations
# toKeep = np.ravel(timeIndex == 0)
# excitations = excitations[toKeep]
# designmatrix = designmatrix[toKeep]

#shuffle
shuffledIndices = np.random.permutation(np.arange(designmatrix.shape[0]))
designmatrix = designmatrix[shuffledIndices]
excitations = excitations[shuffledIndices]

numTrain = int(np.floor(0.8 * excitations.shape[0]))
numVal = excitations.shape[0] - numTrain
trainset = designmatrix[:numTrain]
valset = designmatrix[-numVal:]
trainoutputs = excitations[:numTrain]
valoutputs = excitations[-numVal:]

def readbatch(n, mode = 'train'):
    if mode == 'train':
        dataset = trainset
        outputs = trainoutputs
    else:
        dataset = valset
        outputs = valoutputs
    indices = np.random.choice(dataset.shape[0],size=n)
    return (dataset[indices], np.reshape(outputs[indices], (-1,1)))

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

# Here x and y_ aren't specific values. Rather, they are each a placeholder --
# a value that we'll input when we ask TensorFlow to run a computation.
x = tf.placeholder(tf.float32, shape=[None, trainset.shape[1]])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# We initialized the neuron biases in the second, fourth, and fifth convolutional layers,
# as well as in the fully-connected hidden layers, with the constant 1


keep_prob = tf.placeholder(tf.float32)
fc1, fc1Weight, fc1bias = makeFCLayer(x, 200, 1,'FC1', keep_prob)
# fc2, fc2Weight, fc2bias = makeFCLayer(x, 200, 1,'FC2', keep_prob)

outputWeight = weight_variable([fc1Weight.get_shape()[1].value, 1],'output_weight')
outputbias = bias_variable(1,1,'output_bias')
y = tf.matmul(fc1, outputWeight) + outputbias


# It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variables are uninitialized
# define loss and accuracy functions
with tf.name_scope('Loss_function'):
    #transform eom settings to relative power for loss calcualtion
    def voltage2power(x):
        return (tf.cos(3.1415 + 2 * 3.1415 / 510 * x) + 1) / 2
    clampedy = tf.minimum(255.0, tf.maximum(0.0,y))
    totalSqError = tf.square(voltage2power(clampedy) - voltage2power(y_))
    dataloss = tf.sqrt( tf.reduce_mean(totalSqError))
    tf.summary.scalar('data loss', dataloss)
    # regularization = tf.nn.l2_loss(fc1Weight) + tf.nn.l2_loss(fc2Weight) + tf.nn.l2_loss(fc3Weight)
    # + \   # tf.nn.l2_loss(fc4Weight) + tf.nn.l2_loss(fc5Weight)
    # tf.summary.scalar('weight penalty loss', regularization)
    # weight_decay = 1e-2
    # loss = dataloss + weight_decay*regularization
    loss = dataloss
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# train_step = tf.train.MomentumOptimizer(1e-6, 0.1).minimize(loss)
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
    with open('maitaimodel.csv','wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        variables = [fc1Weight, fc1bias, outputWeight, outputbias]
        for var in variables:
            writer.writerow([var.name])
            array = var.eval(sess)
            if (len(array.shape) == 1):
                array = np.reshape(array,(array.shape[0],1))
            for i in np.arange(array.shape[0]):
                writer.writerow(array[i].tolist())
        #normalization values
        writer.writerow(["DistanceMeans"])
        writer.writerow(distancefeaturesmean)
        writer.writerow(["DistanceSDs"])
        writer.writerow(distancefeaturessd)
        #write test values
        writer.writerow(["TestValues"])
        numTests = 4
        indices = np.random.randint(low=0,high=designmatrix.shape[0],size=(numTests,1))
        for i in indices:
            writer.writerow(np.ravel(designmatrix[i])) #write design mat values
            outputval = sess.run([y], feed_dict={x: designmatrix[i], keep_prob: 1.0})
            writer.writerow(outputval[0][0])
