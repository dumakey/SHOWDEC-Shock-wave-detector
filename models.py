import os
import numpy as np
import tensorflow as tf

def swish(x, beta=1):
    return (x * tf.keras.backend.sigmoid(beta * x))
        
def conv2D_block(X, num_channels, f, p, s, **kwargs):

    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
        dropout = parameters['dropout']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'
        dropout = 0.0

    if p == 'same':
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    elif type(p) == int:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='valid',
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)

    net = tf.keras.layers.BatchNormalization()(net)
    
    if activation == 'leakyrelu':
        rate = 0.2
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.activations.elu(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Dropout(dropout)(net)

    return net

def conv2D_block_with_resnet(X, res_net, num_channels, f, p, s, s_res, **kwargs):

    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
        dropout = parameters['dropout']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'
        dropout = 0.0

    if p == 'same':
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='same',
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
        res = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s_res,padding='same',use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(res_net)
        if res.shape != net.shape:
            res = tf.keras.layers.Resizing(net.shape[1],net.shape[2])(res)
    elif type(p) == int:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
        net = tf.keras.layers.Conv2D(num_channels,kernel_size=f,strides=s,padding='valid',
                                     kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
        res = tf.keras.layers.Conv2D(num_channels,kernel_size=1,strides=s_res,padding='valid',use_bias=True,
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(res_net)

    net = tf.keras.layers.Add()([net,res])
    net = tf.keras.layers.BatchNormalization()(net)

    if activation == 'leakyrelu':
        rate = 0.2
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.activations.elu(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Dropout(dropout)(net)

    return net

def dense_layer(X, units, activation, dropout, l1_reg, l2_reg):

    net = tf.keras.layers.Dense(units=units,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation == 'leakyrelu':
        rate = 0.2
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'elu':
        net = tf.keras.activations.elu(net)
    else:
        net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)

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

def model(image_shape, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    input_shape = (image_shape[1],image_shape[0],3)
    X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=64,f=5,p='same',s=2,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,
                                                                        'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=128,f=3,p='same',s=2,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,
                                                                     'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=128,f=3,p=2,s=1,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'dropout':dropout,
                                                                'activation':activation})
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    
    model = tf.keras.Model(inputs=X_input,outputs=net,name='Model')
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics = [tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.Precision(),
                             tf.keras.metrics.Recall(),
                             ])

    return model


def res_model(image_shape, alpha, l2_reg=0.0, l1_reg=0.0, dropout=0.0, activation='relu'):

    input_shape = (image_shape[1],image_shape[0],3)
    X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=64,f=5,p='same',s=2,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation,
                                                                        'dropout':dropout,'res_net':None})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    res_net_1 = net
    net = conv2D_block(net,num_channels=128,f=3,p='same',s=1,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation,
                                                                     'dropout':dropout})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block_with_resnet(net,res_net_1,num_channels=128,f=3,p='same',s=1,s_res=2,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,
                                                                               'activation':activation,'dropout':dropout})
    net = tf.keras.layers.GlobalMaxPool2D()(net)
    net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)

    model = tf.keras.Model(inputs=X_input,outputs=net,name='Res_Model')
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           ])

    return model