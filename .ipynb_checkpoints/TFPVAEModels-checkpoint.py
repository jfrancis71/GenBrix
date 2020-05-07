import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
tfkl = tfk.layers
tfpl = tfp.layers

base_depth = 32


class FlatVAEModel:
    def __init__( self, latents = 2, q_distribution = tfpl.IndependentNormal, p_distribution_layer_class = tfp.layers.IndependentNormal ):
        self.input_shape = [ 1, 1, 1 ]
        self.latents = latents
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros( self.latents ), scale=1),
            reinterpreted_batch_ndims=1)
        self.q_distribution = q_distribution
        p_distribution_layer = p_distribution_layer_class( self.input_shape )

        self.encoder = \
            tfk.Sequential([
                tfkl.InputLayer(input_shape= self.input_shape ),
                tfkl.Conv2D(base_depth*8, 1, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
                tfkl.Conv2D(base_depth*8, 1, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
                tfkl.Flatten(),
                tfkl.Dense( self.q_distribution.params_size(self.latents),
                   activation=None ),
                self.q_distribution(
                    self.latents,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior) ),
            ])

        self.decoder = \
            tfk.Sequential([
                tfkl.InputLayer(input_shape=[ self.latents ]),
                tfkl.Reshape([1, 1, self.latents]),
                tfkl.Conv2D(base_depth*8, 1, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
                tfkl.Conv2D(base_depth*8, 1, strides=1,
                    padding='same', activation=tf.nn.leaky_relu),
                tfkl.Conv2D(filters=p_distribution_layer.params_size( 1 ), kernel_size=1, strides=1,
                    padding='same', activation=None),
                tfkl.Flatten(),
	        p_distribution_layer
            ])

#Model losely based on: https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_VAE.ipynb
class MNISTVAEModel:
    def __init__( self, latents = 16, q_distribution = tfpl.MultivariateNormalTriL, p_distribution_layer_class = tfp.layers.IndependentBernoulli ):
        self.input_shape = [ 28, 28, 1 ]
        self.latents = latents
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros( self.latents ), scale=1),
            reinterpreted_batch_ndims=1)
        self.q_distribution = q_distribution
        p_distribution_layer = p_distribution_layer_class( self.input_shape )

        self.encoder = \
            tfk.Sequential([
                tfkl.InputLayer(input_shape= self.input_shape ),
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
                tfkl.Dense( self.q_distribution.params_size(self.latents),
                   activation=None ),
                self.q_distribution(
                    self.latents,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior) ),
            ])
        self.decoder = \
            tfk.Sequential([
                tfkl.InputLayer(input_shape=[ self.latents ]),
                tfkl.Reshape([1, 1, self.latents]),
                #note below layer turns into 7x7x....
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
                tfkl.Conv2D(filters=p_distribution_layer.params_size( 1 ), kernel_size=5, strides=1,
                    padding='same', activation=None),
                tfkl.Flatten(),
	        p_distribution_layer
            ])

# 64x64x3 shaped model
# Losely based on: https://github.com/yzwxx/vae-celebA/blob/master/model_vae.py
# Achieved loss of -11,162 on CelebA after 23 epochs on 20,000 images.
# Mean is pretty good, sample is a bit noisy espcially around edges, but quite good
class YZVAEModel():
    def __init__( self, latents = 16, q_distribution = tfpl.MultivariateNormalTriL, p_distribution_layer_class = tfp.layers.IndependentBernoulli ):
        self.input_shape = [ 64, 64, 3 ]
        self.latents = latents
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros( self.latents ), scale=1),
            reinterpreted_batch_ndims=1)
        self.q_distribution = q_distribution
        p_distribution_layer = p_distribution_layer_class( self.input_shape )

        self.encoder = \
            tf.keras.Sequential([
                tfkl.InputLayer( input_shape= self.input_shape ),
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

		tfkl.Dense( self.q_distribution.params_size( self.latents ) ),
		self.q_distribution(
	            self.latents,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior) )
            ])

        self.decoder = \
            tf.keras.Sequential([
                tfkl.InputLayer( input_shape= [ self.latents ] ),
                tfkl.Reshape( [ 1, 1, self.latents ] ),
                tf.keras.layers.Conv2D(
                    filters=64*4*8*8, kernel_size=(1,1), padding='SAME',strides=(1, 1), activation=tf.nn.leaky_relu),
                tf.keras.layers.Reshape( [ 8, 8, 256 ] ),

                tf.keras.layers.Conv2DTranspose(
                    filters=64*4, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu),

                tf.keras.layers.Conv2DTranspose(
                    filters=64*2, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu),

                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(5,5), strides=(2, 2), padding="SAME", activation=tf.nn.leaky_relu),

                tf.keras.layers.Conv2DTranspose(
                    filters=p_distribution_layer.params_size( self.input_shape[2] ), kernel_size=(5,5), strides=(1, 1), padding="SAME", activation=None),
                tfkl.Flatten(),
                p_distribution_layer
            ])
