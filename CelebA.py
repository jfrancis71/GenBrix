
#Achieves 168, generative pretty good. Lots of variety, nearly everything pretty facelike, albeit
#features like ears etc indistinct

import matplotlib.pyplot as plt
import numpy as np
from GenBrix import DataSetUtils
from GenBrix import VariationalAutoencoder as gen_vae

deq = DataSetUtils.read_images( max_no = 20000 )

import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
#    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, 1, 1, 128 ))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VariationalAutoEncoder(tf.keras.Model):

    def __init__(self,
               name='autoencoder',
               **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.vae = gen_vae.YZVAEModel()
        self.encoder = self.vae.inference_net()
        self.decoder = self.vae.generative_net( [ 64, 64, 3 ], 1 )
        self.sampling = Sampling()

    def call(self, inputs):
        enc = self.encoder(inputs)
        reshape_z = tf.keras.layers.Reshape( target_shape = [ 1, 1, 128, 2 ] )( enc )
        z_mean = reshape_z[:,:,:,:,0]
        z_log_var = reshape_z[:,:,:,:,1]
        z = self.sampling( (z_mean, z_log_var) )
        reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
        kl_loss = tf.reduce_mean( - 0.5 * tf.reduce_sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis = [ 1, 2, 3 ] ) )
        self.add_loss(kl_loss)
        return reconstructed

vae = VariationalAutoEncoder()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

def sum_squared_error(y_true, y_pred):
    return tf.reduce_mean( tf.reduce_sum(tf.square(y_pred - y_true), axis=[1, 2, 3 ]) )


vae.compile( optimizer, loss= sum_squared_error )

vae.fit(deq, deq, epochs=60, batch_size=64)

def plot_generate_faces():
    fig, pl = plt.subplots(4,4,figsize=(20,20))
    [ [ pl[x,y].imshow( vae.decoder(  np.random.normal( size = [ 1, 1, 1, 128  ] ) )[0,:,:,:]) for x in range(4) ] for y in range(4) ]

plot_generate_faces()
