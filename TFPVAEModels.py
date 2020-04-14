import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers

base_depth = 32

class MNISTVAEModel:
    def __init__( self ):
        self.latents = 16
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros( self.latents ), scale=1),
            reinterpreted_batch_ndims=1)

    def encoder( self ):
        return tfk.Sequential([
            tfkl.InputLayer(input_shape= [ 28, 28, 1 ] ),
            tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            tfkl.Conv2D(base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(2 * base_depth, 5, strides=1,
                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(2 * base_depth, 5, strides=2,
                padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(4 * self.latents, 7, strides=1,
                padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Flatten(),
            tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(self.latents),
               activation=None),
            tfpl.MultivariateNormalTriL(
                self.latents,
                activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior)),
])
    def decoder( self, distribution ):
        return tfk.Sequential([
            tfkl.InputLayer(input_shape=[ self.latents ]),
            tfkl.Reshape([1, 1, self.latents]),
            tfkl.Conv2DTranspose(2 * base_depth, 7, strides=1,
                         padding='valid', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(2 * base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(2 * base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(base_depth, 5, strides=2,
                         padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2DTranspose(base_depth, 5, strides=1,
                         padding='same', activation=tf.nn.leaky_relu),
            tfkl.Conv2D(filters=distribution[0]*1, kernel_size=5, strides=1,
                padding='same', activation=None),
            tfkl.Reshape( [ 28, 28, 1, distribution[0] ] ),
	    tfpl.DistributionLambda( distribution[1] )
])

# 64x64x3 shaped model
# Achieved loss of -11,162 on CelebA after 23 epochs on 20,000 images.
# Mean is pretty good, sample is a bit noisy espcially around edges, but quite good
class YZVAEModel():
    def __init__( self ):
        self.latents = 16
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros( self.latents ), scale=1),
            reinterpreted_batch_ndims=1)

    def encoder( self ):
        return tf.keras.Sequential([
                tfkl.InputLayer( input_shape= [ 64, 64, 3 ] ),
            tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
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
                tf.keras.layers.ReLU(),

                tfkl.Flatten(),

		tfkl.Dense( tfpl.MultivariateNormalTriL.params_size( self.latents ) ),
		tfpl.MultivariateNormalTriL(
	            self.latents,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))
])

    def decoder( self, distribution ):

        return tf.keras.Sequential([
            tfkl.InputLayer( input_shape= [ self.latents ] ),
            tfkl.Reshape( [ 1, 1, self.latents ] ),
            tf.keras.layers.Conv2D(
                    filters=64*4*8*8, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape( [ 8, 8, 256 ] ),

            tf.keras.layers.Conv2DTranspose(
            filters=64*4, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu),

            tf.keras.layers.Conv2DTranspose(
                    filters=64*2, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu)
,

            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu),

            tf.keras.layers.Conv2DTranspose(
                filters=3*distribution[0], kernel_size=(5,5), strides=(1, 1), padding="SAME", activation=None),
            tfkl.Reshape( [ 64, 64, 3, distribution[0] ] ),
            tfpl.DistributionLambda( distribution[1] )
] )
