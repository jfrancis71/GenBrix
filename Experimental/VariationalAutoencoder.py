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
        return np.random.normal( np.zeros( [ 1, 1, 1, self.latent ] ), np.ones( [ 1, 1, 1, self.latent ] ) ).astype( np.float32 )

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
        return np.random.normal( np.zeros( [ 1, 1, 1, 50 ] ), np.ones( [ 1, 1, 1, 50 ] ) ).astype( np.float32 )

#Model taken from https://github.com/yzwxx/vae-celebA/blob/master/model_vae.py
class YZVAEModel( VAEModel ):
    def __init__( self ):
        super(VAEModel, self).__init__()
        self.trunk_inference = \
            tf.keras.Sequential([

                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
                tf.keras.layers.ReLU(),
            
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2D(
                    filters=512, kernel_size=(5,5), padding='SAME',strides=(2, 2), activation=None),
                tf.keras.layers.ReLU() ] )

        inf_dense_mean = tf.keras.layers.Conv2D(
                filters=512, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None )

        inf_dense_logvar = tf.keras.layers.Conv2D(
                filters=512, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None )


    def inference_net( self ):
        input = tf.keras.layers.Input( [ 64, 64, 3 ] )
        net = self.trunk_inference( input )
        reshaped = tf.keras.layers.Reshape( target_shape = ( 1, 1, 4*4*512 ) )( net )
        mean = tf.keras.layers.Conv2D(
                filters=128, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None )( reshaped )
        logvar = tf.keras.layers.Conv2D(
                filters=128, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None )( reshaped )
        out = tf.stack( [ mean, logvar ], axis=4 )
        reshaped1 = tf.keras.layers.Reshape( target_shape = ( 1, 1, 128*2 ) )( out )
        model = tf.keras.Model( inputs = input, outputs = reshaped1 )
        return model

    def generative_net( self, image_dims, no_distribution_parameters ):

        return tf.keras.Sequential([

            tf.keras.layers.Conv2D(
                    filters=64*4*8*8, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=None),
            tf.keras.layers.Reshape( [ 8, 8, 256 ] ),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(
            filters=64*4, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(
                    filters=64*2, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(
                filters=3*no_distribution_parameters, kernel_size=(5,5), strides=(1, 1), padding="SAME", activation=None) ] )


    def sample_latent( self ):
        return np.random.normal( np.zeros( [ 1, 1, 1, 128 ] ), np.ones( [ 1, 1, 1, 128 ] ) ).astype( np.float32 )



class VariationalAutoencoder(nb.Model):
    def __init__( self, distribution, image_dims, vae_model=DefaultVAEModel() ):
        super(VariationalAutoencoder, self).__init__()
        self.xinference_net = vae_model.inference_net()
        self.xgenerative_net = vae_model.generative_net( image_dims, distribution.no_of_parameters() )
        self.distribution = distribution
        self.vae_model = vae_model
        self.latent_distribution = nb.RealGauss()

    def kl_loss( self, sample_z, z_params ):
        kl_loss = 0.5 * ( -z_params[:,:,:,:,1] + tf.exp( z_params[:,:,:,:,1] ) + z_params[:,:,:,:,0]*z_params[:,:,:,:,0] - 1 )
        return tf.reduce_mean( tf.reduce_sum( kl_loss, axis = [ 1, 2, 3 ] ) )

    def loss( self, samples, logging_context=None, epoch=None ):
        inf = self.xinference_net( samples )
        inf_params = nb.reshape_channel_to_parameters( inf, 2 )
        sample_z = self.latent_distribution.sample( inf )
        
        gen_params = self.xgenerative_net( sample_z  )
        reconstruction_loss = self.distribution.loss( gen_params, samples )
        kl_loss = self.kl_loss( sample_z, inf_params )
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
