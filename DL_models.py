import os
import numpy as np
import tensorflow as tf

def swish(x, beta=1):
    return (x * tf.keras.backend.sigmoid(beta * x))

def conv2D_block(X, num_channels, f, p, s, dropout, **kwargs):
    if kwargs:
        parameters = list(kwargs.values())[0]
        reg_constant = parameters['l2_reg']
        act_fun = parameters['act_fun']
    else:
        reg_constant = 0.0
        act_fun = 'relu'

    if p != 0:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
    else:
        net = X
    net = tf.keras.layers.Conv2D(num_channels, kernel_size=f, strides=s, padding='valid',
                        kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
    net = tf.keras.layers.BatchNormalization()(net)
    
    if act_fun == 'leaky':
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif act_fun == 'linear':
        net = tf.keras.layers.Activation('linear')(net)
    elif act_fun == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
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

def slice_scanner_inception_model(image_shape, reg_constant, alpha, dropout=0.0, pretrained_model=None):

    if pretrained_model == None:
        input_shape = tuple(image_shape.as_list() + [3])
        X_input = tf.keras.layers.Input(shape=input_shape)
        net = conv2D_block(X_input,num_channels=32,f=5,p=0,s=1,dropout=0.0,kwargs={'l2_reg':reg_constant,'act_fun':'swish'})
        net = tf.keras.layers.AvgPool2D(pool_size=3,strides=3)(net)
        net = conv2D_block(net,num_channels=64,f=5,p=0,s=1,dropout=0.0,kwargs={'l2_reg':reg_constant,'act_fun':'swish'})
        net = tf.keras.layers.AvgPool2D(pool_size=3,strides=3)(net)
        net = inception_block(X=net,num_channels=[256,128,64],f=[3,5,9],p=[(1,1),(2,2),(19,25)],s=[(1,1),(1,1),(2,2)],
                              reg=reg_constant)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=50,activation=None,kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=50,activation=None,kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        model = tf.keras.Model(inputs=X_input,outputs=net,name='Inception_SliceScanner')
    else:
        model = pretrained_model

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])

    return model
    
def slice_scanner_lenet_model(image_shape, reg_constant, alpha, dropout=0.0, pretrained_model=None):

    if pretrained_model == None:
        input_shape = tuple(image_shape.as_list() + [3])
        X_input = tf.keras.layers.Input(shape=input_shape)
        net = conv2D_block(X_input,num_channels=6,f=5,p=0,s=1,dropout=0.0,kwargs={'l2_reg':reg_constant,'act_fun':'swish'})
        net = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)(net)
        net = conv2D_block(net,num_channels=16,f=5,p=0,s=1,dropout=0.0,kwargs={'l2_reg':reg_constant,'act_fun':'swish'})
        net = tf.keras.layers.AvgPool2D(pool_size=2,strides=2)(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=120,activation=None,kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=84,activation=None,kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)

        model = tf.keras.Model(inputs=X_input,outputs=net,name='LeNetSliceScanner')
    else:
        model = pretrained_model

    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])

    return model
    
def slice_scanner_conv_model(image_shape, reg_constant, alpha, dropout=0.0, pretrained_model=None):

    if pretrained_model == None:
        input_shape = tuple(image_shape.as_list() + [3])
        X_input = tf.keras.layers.Input(shape=input_shape)
        net = conv2D_block(X_input,num_channels=48,f=11,p=0,s=4,dropout=0.0,kwargs={'l2_reg':reg_constant,'act_fun':'swish'})
        net = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(net)
        net = conv2D_block(net,num_channels=96,f=5,p=2,s=1,dropout=0.0,kwargs={'l2_reg':reg_constant,'act_fun':'swish'})
        net = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(net)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=120,activation=None,kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=84,activation=None,kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation('swish')(net)
        net = tf.keras.layers.Dropout(dropout)(net)
        net = tf.keras.layers.Dense(units=1,activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(reg_constant))(net)
        model = tf.keras.Model(inputs=X_input,outputs=net,name='ConvSliceScanner')
    else:
        model = pretrained_model

    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC()])

    return model
