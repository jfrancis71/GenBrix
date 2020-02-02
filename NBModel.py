
import tensorflow as tf
import numpy as np

class Distribution:
    def no_of_parameters( self ):
        return "unimplemented"
    def loss( self, array, samples ):
        return "unimplemented"
    def sample( self, array1 ):
        return "unimplemented"

class Binary(Distribution):
    def no_of_parameters( self ):
        return 1
    def loss( self, array, samples ):
        tp = tf.broadcast_to( array, samples.shape )
        loss = tf.nn.sigmoid_cross_entropy_with_logits( logits=tp, labels = samples )
        return tf.math.reduce_mean( loss )
    def sample( self, array ):
        return np.random.binomial( 1, tf.math.sigmoid( array ) )

class RealGauss(Distribution):
    def no_of_parameters( self ):
        return 2
    def loss( self, array, samples ):
        reshape = reshape_array( array, samples, 2 )
        loss = -log_normal_pdf( samples, reshape[:,:,:,:,0], reshape[:,:,:,:,1] )
        return tf.math.reduce_mean( loss )
    def sample( self, array ):
        array_shape = tf.shape( array )
        no_image_channels = tf.math.floordiv( array_shape[2], 2 )
        reshape = tf.reshape ( array, [ array_shape[0], array_shape[1], no_image_channels, 2 ] )
        return np.random.normal(
            reshape[:,:,:,0],
            np.sqrt( np.exp( reshape[:,:,:,1] ) ) )

class Discrete(Distribution):
    def no_of_parameters( self ):
        return 10
    def loss( self, array, samples ):
        broad_discrete = reshape_array( array, samples, 10 )
        scale_input = tf.multiply( samples, scale_const )
        rounds = tf.cast( tf.clip_by_value( tf.round( scale_input ), 0, 9 ), tf.int64 )
        cross = tf.nn.sparse_softmax_cross_entropy_with_logits( rounds, broad_discrete )
        mean_cross = tf.math.reduce_mean( cross )
        return mean_cross
    def sample( self, array ):
        array_shape = tf.shape( array )
        no_image_channels = tf.math.floordiv( array_shape[2], 10 )
        reshape = tf.reshape( array, [ array_shape[0], array_shape[1], no_image_channels, 10 ] )
        soft = tf.nn.softmax( reshape )
        return np.apply_along_axis( sample, 3, soft.numpy() )/10

#Reshapes array so that it goes from y*x*(array channels) -> b*y*x*(input channels)*p
#where p is number of parameters of distribution
def reshape_array( array, samples, no_parameters ):
    sample_shape = tf.shape( samples )
    array_shape = tf.shape( array )
    reshape = tf.reshape ( array, [ sample_shape[1], sample_shape[2], sample_shape[3], no_parameters ] )
    return tf.broadcast_to( reshape, [ sample_shape[0], sample_shape[1], sample_shape[2], sample_shape[3], no_parameters ] )

def log_normal_pdf(sample, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
      )

def sample( probs ):
    return np.random.choice( 10, 1, p=probs )[0]

class NBModel:
    def __init__(self, distribution, dims):
        self.distribution = distribution
        dims[2] = dims[2] * distribution.no_of_parameters()
        self.array = tf.Variable(
            np.zeros( dims ).astype('float32') )

    def log_density( self, sample ):
        return self.distribution.loss( self.array, tf.expand_dims( sample, 0 ) )

    def log_densities( self, samples ):
        train_dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(60000).batch(64)
        mean = 0
        count = 0
        for train_x in train_dataset:
            mean += self.distribution.loss( self.array, train_x )
            count += 1
        return mean/count


    def sample( self ):
        return self.distribution.sample( self.array )

    def train( self, samples, no_epoch=10, learning_rate=.01 ):
        train_dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(60000).batch(64)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for epoch in range(no_epoch):
            for train_x in train_dataset:
                self.apply_gradients( train_x, optimizer)
            print( "Epoch", epoch, "Training loss ", tf.reduce_mean( self.log_densities( samples ) ) )

    def apply_gradients( self, samples, optimizer ):
        with tf.GradientTape() as tape:
            loss = self.distribution.loss( self.array, samples )
        gradients = tape.gradient( loss, self.array )
        optimizer.apply_gradients( zip( [ gradients ], [ self.array ] ) )

scale_const = tf.constant( 10.0 )
