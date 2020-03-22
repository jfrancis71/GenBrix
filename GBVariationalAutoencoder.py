
#Achieves 110, generative pretty good. Lots of variety, nearly everything pretty facelike, albeit
#features like ears etc indistinct

import matplotlib.pyplot as plt
import numpy as np
from GenBrix import DataSetUtils
from GenBrix import VAEModels as vae_models
from GenBrix import NBModel as nb

import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__( self, latents ):
        super( Sampling, self).__init__()
        self.latents = latents

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        epsilon = tf.keras.backend.random_normal(shape=(batch, 1, 1, self.latents ))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(tf.keras.Model):

    def __init__(self, vae_model, dimensions, latents ):
        super(VariationalAutoEncoder, self).__init__()
        self.vae = vae_model
        self.encoder = self.vae.inference_net()
        self.decoder = self.vae.generative_net( dimensions, 1 )
        self.sampling = Sampling( latents )
        self.latents = latents

    def kl_loss( self, sample_z, z_params ):
        kl_loss = 0.5 * ( -z_params[:,:,:,:,1] + tf.exp( z_params[:,:,:,:,1] ) + z_params[:,:,:,:,0]*
z_params[:,:,:,:,0] - 1 )
        return tf.reduce_mean( tf.reduce_sum( kl_loss, axis = [ 1, 2, 3 ] ) )


    def call(self, inputs):
        enc = self.encoder(inputs)
        reshape_z = tf.keras.layers.Reshape( target_shape = [ 1, 1, self.latents, 2 ] )( enc )
        z_mean = reshape_z[:,:,:,:,0]
        z_log_var = reshape_z[:,:,:,:,1]
        z = self.sampling( (z_mean, z_log_var) )
        reconstructed = self.decoder(z)
    # Add KL divergence regularization loss.
        kl_loss = self.kl_loss( 0, reshape_z )
        self.add_loss(kl_loss)
        return reconstructed

    def sample( self ):
        z = self.vae.sample_latent()
        return self.decoder( z )

def half_sum_squared_error(y_true, y_pred):
    return 0.5 * tf.reduce_mean( tf.reduce_sum(tf.square(y_pred - y_true), axis=[1, 2, 3 ]) )

def cross_entropy( y_true, y_pred ):
    return tf.reduce_mean( tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits( y_true, y_pred ) ) )

def create_variational_autoencoder_realstd():
    vae = VariationalAutoEncoder( vae_models.YZVAEModel(), [64, 64, 3], 128 )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile( optimizer, loss= half_sum_squared_error )
    return vae

def create_variational_autoencoder_binary():
    vae = VariationalAutoEncoder( vae_models.DefaultVAEModel(), [ 28, 28, 1 ], 64 )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile( optimizer, loss=cross_entropy )
    return vae

