
#All VAEModel's must supply inference_net and generative_net which are functions which build
#a model/layer

import tensorflow as tf
import numpy as np

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



#Model based on https://github.com/yzwxx/vae-celebA/blob/master/model_vae.py
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
                    filters=64*2, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None)
,
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=None),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv2DTranspose(
                filters=3*no_distribution_parameters, kernel_size=(5,5), strides=(1, 1), padding="SAME", activation=None) ] )


    def sample_latent( self ):
        return np.random.normal( np.zeros( [ 1, 1, 1, 128 ] ), np.ones( [ 1, 1, 1, 128 ] ) ).astype( np.float32 )

