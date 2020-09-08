import tensorflow as tf
import numpy as np

def Layer(X,Input,Output,batchNorm=True,Activation='None',Train=True,Dropout=0.2):
    W = tf.Variable(tf.random_normal([Input,Output],stddev=0.01))
    b = tf.Variable(tf.random_normal([Output],stddev=0.01))
    logit = tf.matmul(X,W) + b
    
    if(batchNorm):
        logit = tf.layers.batch_normalization(logit,momentum=0.99)
    
    if(Activation == 'relu'):
        logit = tf.nn.relu(logit)
    elif(Activation == 'leaky_relu'):
        logit = tf.nn.leaky_relu(logit,alpha=0.3)
    elif(Activation == 'softmax'):
        logit = tf.nn.softmax(logit)
    elif(Activation == 'sigmoid'):
        logit = tf.nn.sigmoid(logit)
        
    if(Dropout > 0 and Train==True):
        logit = tf.nn.dropout(logit,Dropout)
    
    return logit
	
def CNNLayer(X,filters,kernelSize,padding='SAME',
             batchNorm=True,Activation='None',
             maxPooling=True,
             Train=True,Dropout=0.2):
    
    chan = X.get_shape().as_list()[3]
    kernelSize.extend([chan,filters])
    weight = tf.Variable(tf.random_normal(kernelSize,stddev=0.01))
    logit = tf.nn.conv2d(X,weight, strides=[1,1,1,1], padding=padding)
    
    if(batchNorm):
        logit = tf.layers.batch_normalization(logit,momentum=0.99)
    
    if(Activation == 'relu'):
        logit = tf.nn.relu(logit)
    elif(Activation == 'leaky_relu'):
        logit = tf.nn.leaky_relu(logit,alpha=0.3)
    elif(Activation == 'softmax'):
        logit = tf.nn.softmax(logit)
    elif(Activation == 'sigmoid'):
        logit = tf.nn.sigmoid(logit)
        
    if(maxPooling == True):
        logit = tf.nn.max_pool(logit,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
    if(Dropout > 0 and Train==True):
        logit = tf.nn.dropout(logit,Dropout)
    
    return logit
	