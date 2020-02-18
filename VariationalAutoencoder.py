import tensorflow as tf
import numpy as np

from GenBrix import NBModel as nb

class VAEModel():

    def generative_net( image_dims, no_of_parameters ):
        return "unimplemented"

    def inference_net():
        return "unimplemented"

    def sample_latent():
        return "unimplemented"

class DefaultVAEModel( VAEModel ):
    def __init__( self, latent=64 ):
        super(VAEModel, self).__init__()
        self.latent = latent

    def generative_net( self, image_dims, no_distribution_parameters ):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D( 
                filters=500, kernel_size=1, padding='SAME', activation='relu' ),
            tf.keras.layers.Conv2D( 
                filters=500, kernel_size=1, padding='SAME', activation='relu' ),
            tf.keras.layers.Conv2D( 
                filters=image_dims[0]*image_dims[1]*image_dims[2]*no_distribution_parameters, kernel_size=1, padding='SAME' ),
            tf.keras.layers.Reshape( target_shape=(image_dims[0],image_dims[1],image_dims[2]*no_distribution_parameters) )
])

    def inference_net( self ):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D( 
                filters=500, kernel_size=1, padding='SAME', activation='relu' ),
            tf.keras.layers.Conv2D( 
                filters=500, kernel_size=1, padding='SAME', activation='relu' ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense( units=self.latent*2, activation=None),
            tf.keras.layers.Reshape( target_shape=(1,1,self.latent*2))
])

    def sample_latent( self ):
        return np.random.normal( np.zeros( [ 1, 1, 1, self.latent ] ), np.ones( [ 1, 1, 1, self.latent ] ) )

#This is a convolution latent variable version of Tensorflow demo example
class ConvVAEModel( VAEModel ):
    def __init__( self ):
        super(VAEModel, self).__init__()

    def inference_net( self ):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding='SAME',strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, padding='SAME',strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense( units=50*2, activation=None),
            tf.keras.layers.Reshape( target_shape=(1,1,50*2))
])

    def generative_net( self, image_dims, no_distribution_parameters ):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=(image_dims[0]//4)*(image_dims[1]//4)*32, kernel_size=1, padding='SAME',strides=(1, 1), activation='relu'),
            tf.keras.layers.Reshape( target_shape=(image_dims[0]//4,image_dims[1]//4,32) ),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=image_dims[2]*no_distribution_parameters, kernel_size=3, strides=(1, 1), padding="SAME", activation=None)
    ])

    def sample_latent( self ):
        return np.random.normal( np.zeros( [ 1, 1, 1, 50 ] ), np.ones( [ 1, 1, 1, 50 ] ) )

class VariationalAutoencoder(nb.Model):
    def __init__( self, distribution, image_dims, vae_model=DefaultVAEModel() ):
        super(VariationalAutoencoder, self).__init__()
        self.xinference_net = vae_model.inference_net()
        self.xgenerative_net = vae_model.generative_net( image_dims, distribution.no_of_parameters() )
        self.distribution = distribution
        self.vae_model = vae_model
        self.latent_distribution = nb.RealGauss()

    def loss( self, samples, logging_context=None, epoch=None ):
        inf = self.xinference_net( samples )
        inf_params = nb.reshape_channel_to_parameters( inf, 2 )
        sample_z = self.latent_distribution.sample( inf )
        
        gen_params = self.xgenerative_net( sample_z  )
        reconstruction_loss = self.distribution.loss( gen_params, samples )
        logpz = nb.log_normal_pdf(sample_z, inf_params[:,:,:,:,0]*0.0, inf_params[:,:,:,:,0]*0. )
        logqz_x = nb.log_normal_pdf(sample_z, inf_params[:,:,:,:,0], inf_params[:,:,:,:,1] )
        kl_loss = logqz_x - logpz
        loss = tf.reduce_mean( reconstruction_loss + kl_loss )
        if logging_context is not None:
            tf.summary.scalar( logging_context+"_kl_loss", kl_loss, step=epoch )
            tf.summary.scalar( logging_context+"_reconstruction_loss", reconstruction_loss, step=epoch )
        return loss
            
    def sample( self, test_z=None ):
        if test_z is None:
            test_z = self.vae_model.sample_latent()
        return self.distribution.sample( self.xgenerative_net( test_z ) )
    
    def get_trainable_variables( self ):
        return self.xinference_net.trainable_variables + self.xgenerative_net.trainable_variables
