#Simple Test harness.
#Will not test for generation performance as too time consuming to be practical.
#Intended as a basic sanity test for the program logic.

from GenBrix import NBModel as nb
from GenBrix import VariationalAutoencoder as vae
from GenBrix import PixelCNN as cnn
from GenBrix import PixelVAE as pvae

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

def test_model( model, model_name, images, test_z, type, no_epoch = 10, learning_rate=.01, logging=False):
    model.train( images, no_epoch, learning_rate, logging )
    print( model_name, ", Log density ", model.log_density( images[0]) )
    if ( test_z is None ):
        sample = model.sample()
    else:
        sample = model.sample( test_z )

    if ( type == 'bin' ):
        sample = sample[0,:,:,0]
        cmap = 'gray'
    else:
        sample = sample[0,:,:,:]
        cmap = None

    plt.imshow( sample, cmap=cmap, vmin=0, vmax=1 )
    plt.show()

test_z = np.random.normal( np.zeros( [ 1, 1, 1, 8 ] ), np.ones( [ 1, 1, 1, 8 ] ) )

def test_nb( image_range=512, no_epoch=10, learning_rate=.0001 ):
    test_model( nb.NBModel( nb.Binary(), [ 28, 28, 1 ] ), "NB Bin", train_bin_images[:image_range], None, 'bin', no_epoch, learning_rate )
    test_model( nb.NBModel( nb.RealGauss(), [ 32, 32, 3 ] ), "NB RealGauss", deq_train_col_images[:image_range], None, 'col', no_epoch, learning_rate )
    test_model( nb.NBModel( nb.Discrete(), [ 32, 32, 3 ] ), "NB Discrete", deq_train_col_images[:image_range], None, 'col', no_epoch, learning_rate )

def test_vae( image_range=512, no_epoch=10, learning_rate=.0001 ):
    test_model( vae.VariationalAutoencoder( nb.Binary(), [ 28, 28, 1 ] ), "VAE Bin", train_bin_images[:image_range], test_z, 'bin', no_epoch, learning_rate )
    test_model( vae.VariationalAutoencoder( nb.RealGauss(), [ 32, 32, 3 ] ), "VAE RealGauss", deq_train_col_images[:image_range], test_z, 'col', no_epoch, learning_rate )
    test_model( vae.VariationalAutoencoder( nb.Discrete(), [ 32, 32, 3 ] ), "VAE Dscrete", deq_train_col_images[:image_range], test_z, 'col', no_epoch, learning_rate )


def test_cnn( image_range=512, no_epoch=10, learning_rate=.0001 ):
    test_model( cnn.PixelCNN( nb.Binary(), [ 28, 28, 1] ), "CNN Bin", train_bin_images[:image_range], None, 'bin', no_epoch, learning_rate )
    test_model( cnn.PixelCNN( nb.RealGauss(), [ 32, 32, 3 ] ), "CNN RealGauss", deq_train_col_images[:image_range], None, 'col', no_epoch, learning_rate )
    test_model( cnn.PixelCNN( nb.Discrete(), [ 32, 32, 3 ] ), "CNN Discrete", deq_train_col_images[:image_range], None, 'col', no_epoch, learning_rate )

def test_pixelvae( image_range=512, no_epoch=10, learning_rate=.0001 ):
    test_model( pvae.PixelVAE( nb.Binary(), [ 28, 28, 1 ] ), "PixelVAE Bin", train_bin_images[:image_range], test_z, 'bin', no_epoch, learning_rate )
    test_model( pvae.PixelVAE( nb.RealGauss(), [ 32, 32, 3 ] ), "PixelVAE RealGauss", deq_train_col_images[:image_range], test_z, 'col', no_epoch, learning_rate )
    test_model( pvae.PixelVAE( nb.Discrete(), [ 32, 32, 3 ] ), "PixelVAE Discrete", deq_train_col_images[:image_range], test_z, 'col', no_epoch, learning_rate )



def test( image_range=512, no_epoch=2, learning_rate=.0001 ):
    test_nb( image_range, no_epoch, learning_rate )
    test_vae( image_range, no_epoch, learning_rate )
    test_cnn( image_range, no_epoch, learning_rate )
    test_pixelvae( image_range, no_epoch, learning_rate )
