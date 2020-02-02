
from GenBrix1 import NBModel as nb
#from GenBrix1 import GBVaE as GBVaE

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython import display

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')/255
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')/255.

# Binarization
train_bin_images = train_images
test_bin_images = test_images
train_bin_images[train_images >= .5] = 1.
train_bin_images[train_images < .5] = 0.
test_bin_images[test_images >= .5] = 1.
test_bin_images[test_images < .5] = 0.

deq_train_images=train_images + np.random.normal( 0, .05, [ 60000, 28, 28, 1 ]).astype(np.float32)

( train_col_images , _ ), _ = tf.keras.datasets.cifar10.load_data()
train_col_images = train_col_images.reshape(train_col_images.shape[0], 32, 32, 3).astype('float32')/255

deq_train_col_images=train_col_images + np.random.normal( 0, .05, [ 50000, 32, 32, 3 ]).astype(np.float32)

def test_nb_bin( images, dims, no_epoch = 10, learning_rate=.01):
    nbmodel = nb.NBModel( nb.Binary(), dims )
    nbmodel.train( images, no_epoch, learning_rate )
    print( "GB Binary NBModel Log density ", nbmodel.log_density( images[0]) )
    plt.imshow( nbmodel.sample()[:,:,0], cmap='gray', vmin=0, vmax=1 )
    plt.show()

def test_nb_real_gauss( images, dims, no_epoch = 10, learning_rate=.01):
    nbmodel = nb.NBModel( nb.RealGauss(), dims )
    nbmodel.train( images, no_epoch, learning_rate )
    print( "GB RealGauss NBModel Log density ", nbmodel.log_density( images[0]) )
    plt.imshow( nbmodel.sample()[:,:,0], vmin=0, vmax=1 )
    plt.show()

def test_nb_discrete( images, dims, no_epoch = 10, learning_rate=.01):
    nbmodel = nb.NBModel( nb.Discrete(), dims )
    nbmodel.train( images, no_epoch, learning_rate )
    print( "GB Discrete NBModel Log density ", nbmodel.log_density( images[0]) )
    plt.imshow( nbmodel.sample()[:,:,0], vmin=0, vmax=1 )
    plt.show()

def test( no_epoch=10, learning_rate=.01 ):
   test_nb_bin( train_bin_images, [ 28, 28, 1 ], no_epoch, learning_rate )
   test_nb_real_gauss( deq_train_col_images, [ 32, 32, 3 ], no_epoch, learning_rate )
   test_nb_discrete( deq_train_col_images, [ 32, 32, 3 ], no_epoch, learning_rate )
