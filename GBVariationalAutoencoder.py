#CelebA, 20,000 images over 60 epochs. Batch size 64
#RealStd Achieves 11403, generative pretty good. Lots of variety, nearly everything pretty facelike,
#albeit features like ears etc indistinct
#Discrete, not currently used
#Discrete achieves 9841, specular, stongly shape-like but mostly not face-like
#Incidentally if you divide reconstruction loss on Discrete by 10 you get 1277 loss, and much
#more facelike images.

import matplotlib.pyplot as plt
import numpy as np
from GenBrix import DataSetUtils
from GenBrix import VAEModels as vae_models
from GenBrix import NBModel as nb

import tensorflow as tf
import tensorflow_probability as tfp

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

    def __init__(self, vae_model, dimensions, latents, distribution, distribution_no_parameters ):
        super(VariationalAutoEncoder, self).__init__()
        self.vae = vae_model
        self.encoder = self.vae.inference_net()
        self.decoder = self.vae.generative_net( dimensions, distribution, distribution_no_parameters )
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
        return self.decoder( z ).sample()

scale_const = tf.constant( 10.0 )

def discrete_loss( y_true, y_pred ):
    logits_parameters_output = nb.reshape_channel_to_parameters( y_pred, 10 )
    scale_input = tf.multiply( y_true, scale_const )
    rounds = tf.cast( tf.clip_by_value( tf.round( scale_input ), 0, 9 ), tf.int64 )
    cross = tf.nn.sparse_softmax_cross_entropy_with_logits( rounds, logits_parameters_output )
    loss = tf.math.reduce_mean( tf.math.reduce_sum( cross, axis = [ 1,2, 3 ] ) )
    return loss

def negative_log_likelihood( x, rv_x ):
    return tf.reduce_mean( tf.reduce_sum( -rv_x.log_prob( x ), axis = [ 1, 2, 3 ] ) )

def create_variational_autoencoder_realstd():
    distribution = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.Normal( loc=t, scale=1 ))
    vae = VariationalAutoEncoder( vae_models.YZVAEModel(), [64, 64, 3], 128, distribution, 1 )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile( optimizer, loss=negative_log_likelihood )
    return vae

def create_variational_autoencoder_discrete():
    vae = VariationalAutoEncoder( vae_models.YZVAEModel(), [64, 64, 3], 128, 10 )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile( optimizer, loss=discrete_loss  )
    return vae

def create_variational_autoencoder_binary():
    distribution = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.Bernoulli( logits=t ) )
    vae = VariationalAutoEncoder( vae_models.DefaultVAEModel(), [ 28, 28, 1 ], 64, distribution, 1 )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile( optimizer, loss=negative_log_likelihood )
    return vae
