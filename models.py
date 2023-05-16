import os
import numpy as np
import tensorflow as tf

def swish(x, beta=1):
    return (x * tf.keras.backend.sigmoid(beta * x))

def f1_score(y_true, y_pred):

    def recall_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true*y_pred,0,1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true,0,1)))
        recall = true_positives/(possible_positives + tf.keras.backend.epsilon())

        return recall

    def precision_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true*y_pred,0,1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred,0,1)))
        precision = true_positives/(predicted_positives + tf.keras.backend.epsilon())

        return precision

    precision = precision_m(y_true,y_pred)
    recall = recall_m(y_true,y_pred)

    return 2 * ((precision*recall)/(precision + recall + tf.keras.backend.epsilon()))

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true*y_pred,0,1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true,0,1)))
    recall = true_positives/(possible_positives + tf.keras.backend.epsilon())

    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true*y_pred,0,1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred,0,1)))
    precision = true_positives/(predicted_positives + tf.keras.backend.epsilon())

    return precision
        
def conv2D_block(X, num_channels, f, p, s, dropout, **kwargs):

    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    if p != 0:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
    else:
        net = X
    net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='valid',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    
    if activation == 'leakyrelu':
        rate = 0.1
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU()(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net


def inception_block(X, num_channels, f, p, s, reg):

    net = []
    for i in range(num_channels.__len__()):
        padded = tf.keras.layers.ZeroPadding2D((p[i][0],p[i][1]))(X)
        net.append(tf.keras.layers.Conv2D(num_channels[i],kernel_size=f[i],strides=(s[i][0],s[i][1]),padding='valid',
                   kernel_regularizer=tf.keras.regularizers.l2(reg))(padded))
    net.append(tf.keras.layers.MaxPool2D(pool_size=2, padding='same', strides=1)(X))

    net = tf.keras.layers.Concatenate(axis=-1)(net)

    return net

def get_padding(f,s,nin,nout):

    padding = []
    for i in range(f.__len__()):
        p = int(np.floor(0.5 * ((nout-1)*s[i] + f[i] - nin)))
        nchout = int(np.floor((nin + 2*p - f[i])/s[i] + 1))
        if nchout != nout:
            padding.append(p+1)
        else:
            padding.append(p)

    return padding

def slice_scanner_inception_model(image_shape, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    input_shape = tuple(image_shape.as_list() + [3])
    X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=32,f=5,p=0,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                                   'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=3,strides=3)(net)
    net = conv2D_block(net,num_channels=64,f=5,p=0,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                               'activation':activation})
    net = tf.keras.layers.AvgPool2D(pool_size=3,strides=3)(net)
    net = inception_block(X=net,num_channels=[256,128,64],f=[3,5,9],p=[(1,1),(2,2),(19,25)],s=[(1,1),(1,1),(2,2)],
                          reg=l2_reg)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation != 'leakyrelu':
        net = tf.keras.layers.Activation(activation)(net)
    else:
        net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=50,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation != 'leakyrelu':
        net = tf.keras.layers.Activation(activation)(net)
    else:
        net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=50,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation != 'leakyrelu':
        net = tf.keras.layers.Activation(activation)(net)
    else:
        net = tf.keras.layers.LeakyReLU()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    model = tf.keras.Model(inputs=X_input,outputs=net,name='Inception_SliceScanner')

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(),
                      tf.keras.metrics.Precision(),
                      tf.keras.metrics.Recall(),
                      tf.keras.metrics.AUC()])

    return model
    
def slice_scanner_simple_cnn_model(image_shape, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    input_shape = tuple(image_shape.as_list() + [3])
    X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=64,f=5,p=0,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                                  'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    #net = conv2D_block(net,num_channels=64,f=3,p=5,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
    #                                                                              'activation':activation})
    #net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    #net = conv2D_block(net,num_channels=128,f=5,p=0,s=2,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
    #                                                                              'activation':activation})
    #net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=1024,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation == 'leakyrelu':
        rate = 0.1
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU()(net)
    else:
        net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)
    net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    
    model = tf.keras.Model(inputs=X_input,outputs=net,name='CNNScanner')
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics = [tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.Precision(),
                             tf.keras.metrics.Recall(),
                             ])

    return model