import os
import pickle
import keras
import numpy as np
import sys
import argparse
import tensorflow as tf
import keras.backend as K

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Input, Activation, Conv2D, MaxPool2D, AveragePooling2D, Dropout, Flatten, Layer, ZeroPadding2D, Add, Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, Callback
from keras.backend.tensorflow_backend import set_session
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
IMG_SHAPE = (32, 32, 3)
WEIGHT_DECAY = 0.0005
FEATURES_DIM = 128


from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2

def expand_conv(init, base, k, stride):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    shortcut  = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(init)
    shortcut  = Activation('relu')(shortcut)
    
    x = Convolution2D(base * k, (3, 3), strides=stride, padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(base * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    
    shortcut = Convolution2D(base * k, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal', use_bias=False)(shortcut)
    m = Add()([x, shortcut])

    return m


def conv_block(input, n, stride, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(n * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    m = Add()([init, x])
    return m

def wide_residual_network(input_dim, nb_classes=10, N=2, k=1, dropout=0.0, optimizer = 'adam', summary = True):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    ip = Input(shape=input_dim)

    x = ZeroPadding2D((1, 1))(ip)
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(x)

    nb_conv = 4

    x = expand_conv(x, 16, k, stride=(1,1))

    for i in range(N - 1):
        x = conv_block(x, n=16, stride=(1,1), k=k, dropout=dropout)
        nb_conv += 2

    x = expand_conv(x, 32, k, stride=(2,2))

    for i in range(N - 1):
        x = conv_block(x, n=32, stride=(2,2), k=k, dropout=dropout)
        nb_conv += 2

    x = expand_conv(x, 64, k, stride=(2,2))

    for i in range(N - 1):
        x = conv_block(x, n=64, stride=(2,2), k=k, dropout=dropout)
        nb_conv += 2

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model

def svd_orthonormal(shape):
    # Orthonorm init code is taked from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def get_activations(model, layer, X_batch):
    intermediate_layer_model = Model(
        inputs=model.get_input_at(0),
        outputs=layer.get_output_at(0)
    )
    activations = intermediate_layer_model.predict(X_batch)
    return activations


def LSUVinit(model, batch, verbose=True, margin=0.1, max_iter=10):
    # only these layer classes considered for LSUV initialization; add more if needed
    classes_to_consider = (Dense, Conv2D)

    needed_variance = 1.0

    layers_inintialized = 0
    for layer in model.layers:
        if verbose:
            print(layer.name)
        if not isinstance(layer, classes_to_consider):
            continue
        # avoid small layers where activation variance close to zero, esp. for small batches
        if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
            if verbose:
                print(layer.name, 'too small')
            continue
        if verbose:
            print('LSUV initializing', layer.name)

        layers_inintialized += 1
        weights_and_biases = layer.get_weights()
        weights_and_biases[0] = svd_orthonormal(weights_and_biases[0].shape)
        layer.set_weights(weights_and_biases)
        activations = get_activations(model, layer, batch)
        variance = np.var(activations)
        iteration = 0
        if verbose:
            print(variance)
        while abs(needed_variance - variance) > margin:
            if np.abs(np.sqrt(variance)) < 1e-7:
                # avoid zero division
                break

            weights_and_biases = layer.get_weights()
            weights_and_biases[0] /= np.sqrt(variance) / \
                np.sqrt(needed_variance)
            layer.set_weights(weights_and_biases)
            activations = get_activations(model, layer, batch)
            variance = np.var(activations)

            iteration += 1
            if verbose:
                print(variance)
            if iteration >= max_iter:
                break
    if verbose:
        print('LSUV: total layers initialized', layers_inintialized)
    return model


class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, features_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.features_dim = features_dim

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, self.features_dim),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is NxFEATURES_DIM, x[1] is Nx10 onehot, self.centers is 10xFEATURES_DIM
        delta_centers = K.dot(K.transpose(
            x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10xFEATURES_DIM
        center_counts = K.sum(K.transpose(
            x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers

        self.add_update((self.centers, new_centers), x)
        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result  # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base +
               '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b',
               padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c',
               padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1',
                        padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(
        axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def resnet_like(input_shape=(182, 182, 3), classes=6, optimizer = 'adam'):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (5, 5), strides=(2, 2), name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    # X = MaxPool2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(
        X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(
        X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(
        X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=1)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # # # Stage 5 (≈3 lines)
    # X = convolutional_block(
    #     X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line)
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='my_resnet')
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics= ['accuracy'])
    model.summary()
    return model


def conv_factory(input, filters, ksize, padding = 'same', pooling = None, pool_size = 3, activation = 'relu', dropout = 0.5):

    def frac_max_pool(x):
        return tf.nn.fractional_max_pool(x, [1.0, 1.44, 1.44, 1.0])[0]

    inner = Conv2D(filters, kernel_size = ksize, padding = padding, kernel_regularizer = l2(WEIGHT_DECAY))(input)
    if pooling == 'mp':
        inner = MaxPool2D(pool_size = pool_size, strides = 2)(inner)
    elif pooling == 'fmp':
        inner = Lambda(frac_max_pool)(inner)
    else:
        pass

    inner = Activation(activation=activation)(inner)
    inner = BatchNormalization()(inner)
    inner = Dropout(dropout)(inner)
    return inner


def base_model(activation='relu', summary=True, optimizer='adam', width = 32, depth = 3):
    inp = Input(shape=(IMG_SHAPE))
    inner = conv_factory(inp, width, 5, pooling=None,
                         activation=activation, dropout=0.25)
    inner = conv_factory(inner, width, 5, pooling='mp',
                         activation=activation, dropout=0.25)

    for d in range(depth):
        inner = conv_factory(inner, width*(d+2), 5, pooling=None,
                            activation=activation, dropout=0.25)
        inner = conv_factory(inner, width*(d+2), 5, pooling='mp',
                            activation=activation, dropout=0.25)

    embedding = Flatten(name='embedding')(inner)
    out = Dense(512, activation=activation)(embedding)
    out = Dropout(rate=0.25)(out)
    out = Dense(10, name='output')(out)
    out = Activation(activation='softmax')(out)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model

def cnn_fmp(activation='relu', summary=True, optimizer='adam', width = 32, depth = 5):
    inp = Input(shape=(IMG_SHAPE))
    inner = conv_factory(inp, 32, 5, pooling=None,
                         activation=activation, dropout=0.25)
    
    for d in range(depth):
        inner = conv_factory(inner, width * (d+1), 5, pooling='fmp',
                            activation=activation, dropout=0.25)

    embedding = Flatten(name='embedding')(inner)
    out = Dense(512, activation=activation)(embedding)
    out = Dropout(rate=0.25)(out)
    out = Dense(10, name='output')(out)
    out = Activation(activation='softmax')(out)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    if summary:
        model.summary()
    return model

def centerise(model):
    main_inp = model.input
    embed = model.layers[-2].output
    features_dim = model.layers[-2].output_shape[-1]
    nb_classes = model.layers[-1].output_shape[-1]
    aux_inp = Input((nb_classes, ))
    side_out = CenterLossLayer(
        alpha=0.5, features_dim=features_dim, name='centerlosslayer')([embed, aux_inp])
    center_model = Model(inputs=[main_inp, aux_inp],
                         outputs=[model.output, side_out])
    center_model.summary()
    return center_model

def load_data(normalize = True):
    print("Loading data ...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=97)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    # y_valid = keras.utils.to_categorical(y_valid)
    if normalize:
        # mean = np.mean(X_train,axis=(0,1,2,3))
        # std = np.std(X_train,axis=(0,1,2,3))
        # X_train = (X_train-mean)/(std+1e-7)
        # X_test = (X_test-mean)/(std+1e-7)
        # X_valid = (X_valid-mean)/(std+1e-7)
        X_train = X_train.astype(np.float32)/255.
        X_test  = X_test.astype(np.float32)/255.

    return X_train, X_test, y_train, y_test


def load_weights(model, name):
    print("Loading weights ...")
    if os.path.isfile("{}.h5".format(name)):
        model.load_weights("{}.h5".format(name))
    else:
        print("No pretrained weights")


def schedule(epoch_idx):
    if (epoch_idx + 1) < 60:
        return 0.01
    elif (epoch_idx + 1) < 120:
        return 0.005 # lr_decay_ratio = 0.2
    return 0.005

def build_model(args):
    opt = RMSprop(lr=0.001, decay=1e-6)
    # opt = Adam()

    model_type = args.type
    if model_type == 'wrn':
        model = wide_residual_network(
            IMG_SHAPE, 10, args.wrn_d, args.wrn_w, 0.5, optimizer=opt)
    elif model_type == 'base':
        model = base_model(optimizer=opt)
    elif model_type == 'fmp':
        model = cnn_fmp(optimizer=opt)
    elif model_type == 'resnet':
        model = resnet_like(IMG_SHAPE, 10, optimizer=opt)

    return model 

def main(args):
    try:
        batch_size = args.batch_size
        epochs = args.epochs
        model_type = args.type
        if model_type == 'wrn':
            model_name = '_'.join([args.type, str(args.wrn_d), str(args.wrn_w), str(args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        elif model_type == 'fmp':
            model_name = '_'.join([args.type, str(args.fmp_d), str(args.fmp_w), str(args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        elif model_type == 'base':
            model_name = '_'.join([args.type, str(args.base_d), str(args.base_w), str(args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        elif model_type == 'resnet':
            model_name = '_'.join([args.type, str(args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        
        use_cyclic = args.cyclic
        use_center = args.center
        X_train, X_test, y_train, y_test = load_data(args.normalize)
        

        # As a sanity check, we print out the size of the training and test data.
        print('Training data shape: ', X_train.shape)
        print('Training labels shape: ', y_train.shape)
        # print('Validation data shape: ', X_valid.shape)
        # print('Validation labels shape: ', y_valid.shape)
        print('Test data shape: ', X_test.shape)
        print('Test labels shape: ', y_test.shape)

        print("Building model ...")
        early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        if use_cyclic:
            print("*using cyclical lr")
            lr_schedule = CyclicLR(base_lr=0.001, max_lr=0.006,
                                   step_size=2000., mode='exp_range',
                                   gamma=0.99994)
        else:
            lr_schedule = LearningRateScheduler(schedule)

        save_checkpoint = ModelCheckpoint("{}.h5".format(
            model_name), monitor="val_acc", save_best_only=True, mode='max')

        model = build_model(args)

        if args.lsuv:
            model = LSUVinit(model, X_train[:batch_size])
            
        load_weights(model, model_name)
        
        if use_center:
            print("*using center loss")
            center_model = centerise(model)
            center_model.summary()

            model.compile(optimizer=opt,
                          loss=[categorical_crossentropy, zero_loss],
                          loss_weights=[10, 0.001],
                          metrics=['accuracy'])

            dummy = np.zeros((X_train.shape[0], 1))
            dummy_val = np.zeros((X_test.shape[0], 1))

            print("Start training ...")
            save_checkpoint = ModelCheckpoint("{}.h5".format(
                model_name), monitor="val_main_out_acc", save_best_only=True, mode='max')
            early_stop = EarlyStopping(
                monitor='val_main_out_loss', patience=50, verbose=1)
            history = model.fit(x=[X_train, y_train], y=[y_train, dummy], validation_data=([X_test, y_test], [y_test, dummy_val]),
                                batch_size=batch_size, verbose=1, callbacks=[early_stop, lr_schedule, save_checkpoint], epochs=epochs)
        else:
            print("Start training ...")
            if args.augment:
                datagen = ImageDataGenerator(
                                            featurewise_center=False,  # set input mean to 0 over the dataset
                                            samplewise_center=False,  # set each sample mean to 0
                                            featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                            samplewise_std_normalization=False,  # divide each input by its std
                                            zca_whitening=False,  # apply ZCA whitening
                                            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                            horizontal_flip=True,  # randomly flip images
                                            vertical_flip=False)  # randomly flip images
                datagen.fit(X_train)
                history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
                                                steps_per_epoch = X_train.shape[0] // batch_size, epochs= epochs,\
                                                verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop, lr_schedule, save_checkpoint])
            else:
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                                    batch_size=batch_size, verbose=1, callbacks=[early_stop, lr_schedule, save_checkpoint])
            save_history(history.history, model_name)

        evaluate(args, model_name, model)

    except KeyboardInterrupt:
        print("Terminating ...")
        evaluate(args, model_name, model)


def save_history(history, model_name):
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    print("Saving training history ...")
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']
    
    plt.figure()
    plt.subplot(121)
    plt.xlabel("Ep")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(loss)), loss, c = 'C0', label = 'loss')
    plt.plot(np.arange(len(val_loss)), val_loss, c = 'C1', label = 'val_loss')
    plt.legend()

    plt.subplot(122)
    plt.xlabel("Ep")
    plt.ylabel("Acc")
    plt.plot(np.arange(len(acc)), acc, c = 'C0', label = 'acc')
    plt.plot(np.arange(len(val_acc)), val_acc, c = 'C1', label = 'val_acc')
    plt.legend()

    plt.savefig(model_name)

def eval(X, y, mode, model):
    score = model.evaluate(X, y, batch_size=args.batch_size, verbose=1)
    print("{} accuracy: {}%".format(mode, score[1]*100))

def evaluate(args, model_name, model = None):
    print("Evaluating ...")
    X_train, X_test, y_train, y_test = load_data(args.normalize)
    if model is None:
        model = build_model(args)
    load_weights(model, model_name)

    if not args.center:
        eval(X_train, y_train, 'train', model)
        # eval(X_valid, y_valid, 'valid')
        eval(X_test, y_test, 'test', model)
    else:
        dummy = np.zeros((X_train.shape[0], 1))
        dummy_val = np.zeros((X_test.shape[0], 1))
        score = model.evaluate([X_train, y_train], [y_train, dummy])
        print("Training score: ", score)
        score = model.evaluate([X_test, y_test], [y_test, dummy_val])
        print("Testing score: ", score)
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,
                        help='batch_size', default=32)
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train', default=100)
    parser.add_argument('--cyclic', type=int,
                        help='whether to use cyclical learning rate', default=0)
    parser.add_argument('--center', type=int,
                        help='whether to use center loss', default=0)

    parser.add_argument('--type', type=str, default='base')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--augment', type=int, default=0)
    parser.add_argument('--lsuv', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--wrn_d', type=int, default=4)
    parser.add_argument('--wrn_w', type=int, default=4)
    parser.add_argument('--base_d', type=int, default=3)
    parser.add_argument('--base_w', type=int, default=32)
    parser.add_argument('--fmp_d', type=int, default=5)
    parser.add_argument('--fmp_w', type=int, default=32)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print("Arguments: epochs {}, batch_size {}, use_cyclic {}, mode {}, type {},  user_center {}, augment {}, lsuv {}, normalize {}".format(
        args.epochs, args.batch_size, args.cyclic, args.mode, args.type, args.center, args.augment, args.lsuv, args.normalize))
    if args.mode == 'train':
        main(args)
    elif args.mode == 'eval':
        model_type = args.type
        if model_type == 'wrn':
            model_name = '_'.join([args.type, str(args.wrn_d), str(args.wrn_w), str(
                args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        elif model_type == 'fmp':
            model_name = '_'.join([args.type, str(args.fmp_d), str(args.fmp_w), str(
                args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        elif model_type == 'base':
            model_name = '_'.join([args.type, str(args.base_d), str(args.base_w), str(
                args.normalize), str(args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])
        elif model_type == 'resnet':
            model_name = '_'.join([args.type, str(args.normalize), str(
                args.augment), str(args.cyclic), str(args.lsuv), str(args.center)])

        print(model_name)
        evaluate(args = args, model_name=model_name)

    elif args.mode == 'summary':
        model = build_model(args)
        model.summary()