import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import reparameterization

tfd = tfp.distributions

class Discrete(tfd.Distribution):
    def __init__( self, logits ):
        super( Discrete, self ).__init__(
            dtype=tf.float32,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True)
        self.dist = tfd.Categorical( logits = logits )
        
    def _sample_n( self, n, seed = None ):
        return tf.cast( self.dist.sample( n, seed ), tf.float32 ) * 0.1
    
    def _log_prob( self, samples ):
        #do we need to round?
        return self.dist.log_prob( tf.round( samples*9.0 ) )

def discrete_distribution_fn( t, input_shape ):
    return tfd.Independent(
        Discrete( logits=tf.keras.layers.Reshape( input_shape + [10] )( t ) ), reinterpreted_batch_ndims=3 )

class IndependentDiscrete( tfp.layers.DistributionLambda ):
    def __init__( self, input_shape ):
        super( IndependentDiscrete, self ).__init__(
            make_distribution_fn= lambda t: discrete_distribution_fn( t, input_shape ) )
    
    def params_size( self, channels ):
        return channels*10
