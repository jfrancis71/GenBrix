
#Achieves .0103, generative pretty good. Lots of variety, nearly everything strongly facelike, albeit
#features like ears etc indistinct

import matplotlib.pyplot as plt
import csv
from PIL import Image
import glob as gl
import numpy as np

files = gl.glob("/home/julian/ImageDataSets/CelebA/img_align_celeba/*.jpg")

def read_cropped_image( filename ):
    f = Image.open( filename )
    crop =  f.crop( (15,40,15+148-1,40+148-1))
    newsize = crop.resize( (64,64 ) )
    return newsize

lsamples = [ np.asarray(read_cropped_image( filename ) ) for filename in files[:20000] ]

samples = np.array( lsamples ).astype( np.float32 )

deq = samples/256. + np.random.uniform( low=-0.01,high=0.01, size=[20000,64,64,3]).astype( np.float32)

import tensorflow as tf

inference_trunk = tf.keras.Sequential([

    tf.keras.layers.Conv2D(
        filters=64, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(
        filters=128, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(
        filters=256, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(
        filters=512, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Reshape( target_shape = ( 1, 1, 4*4*512 ) ) ] )

class Sampling(tf.keras.layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
#    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, 1, 1, 128 ))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
  """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

  def __init__(self,
               name='encoder',
               **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs)
    self.trunk = inference_trunk
    self.mean = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None )
    self.log_var = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None )
#    self.sampling = Sampling()

  def call(self, inputs):
    x = self.trunk(inputs)
    z_mean = self.mean(x)
    z_log_var = self.log_var(x)
#    z = self.sampling((z_mean, z_log_var))
    return z_mean, z_log_var

generative_net = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters=64*4*8*8, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None),
    tf.keras.layers.Reshape( [ 8, 8, 256 ] ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=64*4, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=64*2, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2DTranspose(
        filters=3*1, kernel_size=(5,5), strides=(1, 1), padding="SAME", activation=None) ] )

class VariationalAutoEncoder(tf.keras.Model):

    def __init__(self,
               name='autoencoder',
               **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder()
        self.decoder = generative_net
        self.sampling = Sampling()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling( (z_mean, z_log_var) )
        reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss*0.005)
        return reconstructed

vae = VariationalAutoEncoder()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

vae.fit(deq, deq, epochs=60, batch_size=64)

def plot_generate_faces():
    fig, pl = plt.subplots(4,4,figsize=(20,20))
    [ [ pl[x,y].imshow( vae.decoder(  np.random.normal( size = [ 1, 1, 1, 128  ] ) )[0,:,:,:]) for x in range(4) ] for y in range(4) ]

plot_generate_faces()
