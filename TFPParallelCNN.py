#Very simple ParallelCNN, initial draft, specialised for MNIST
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import reparameterization
import numpy as np

tfd = tfp.distributions


def generate_pixel_groups( height, width ):#1 means predict on this iteration
    pixel_groups = np.zeros( [ 4, height, width ] )
    pixel_groups[0,::2,::2] = 1
    pixel_groups[1,1::2,1::2] = 1
    pixel_groups[2,::2,1::2] = 1
    pixel_groups[3,1::2,0::2] = 1
    return pixel_groups

def generate_pixel_channel_groups( dims ):
    pixel_channel_groups = np.zeros( [ 4, dims[2], dims[0], dims[1], dims[2] ])
    pixel_groups = generate_pixel_groups( dims[0], dims[1] )
    for p in range(4):
        for ch in range(dims[2]):
            pixel_channel_groups[p,ch,:,:,ch] = pixel_groups[p,:,:]
    pixel_channel_groups = pixel_channel_groups.reshape( [ dims[2]*4, dims[0], dims[1], dims[2] ] )
    return pixel_channel_groups

#1 means you are allowed to see this, 0 means must be blocked
def generate_information_masks( dims ):
    pixel_channel_groups = generate_pixel_channel_groups( dims )
    information_masks = np.array( [ np.sum( pixel_channel_groups[:x], axis=0 ) if x > 0 else np.zeros( [ dims[0], dims[1], dims[2] ] ) for x in range(4*dims[2]) ] )
    return information_masks

def create_parallelcnns( dims ):
    return [ tf.keras.Sequential([
    tf.keras.layers.Conv2D(
              filters=16, kernel_size=3, padding='SAME',strides=(1, 1), activation='tanh'),
    tf.keras.layers.Conv2D(
              filters=16, kernel_size=1, padding='SAME',strides=(1, 1), activation='tanh'),
    tf.keras.layers.Conv2D(
              filters=dims[2]*1, kernel_size=1, padding='SAME',strides=(1, 1), activation=None),
    tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfd.Bernoulli(
            logits=t) )
]) for x in range(4*dims[2]) ]

class PositionVariance( tf.keras.layers.Layer ):
    def __init__( self ):
        super( PositionVariance, self ).__init__()
        self.position = tf.Variable( np.zeros( [ 28, 28, 1 ] ).astype( np.float32 ) )

    def call( self, input ):
        broadcast_shape = tf.where([True, False, False, False],
            tf.shape( input ), [0, 28, 28, 1 ] )
        broadcast_position = tf.broadcast_to( self.position, broadcast_shape )
        return tf.concat( [ input, broadcast_position ], axis=3 )

class ParallelCNN( tfp.distributions.Distribution ):

    def __init__( self, dims ):
        super(ParallelCNN, self).__init__(
            dtype=tf.float32,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=False,
            allow_nan_stats=True )
        self.parallelcnns = create_parallelcnns( dims )
        self.pixel_channel_groups = generate_pixel_channel_groups( dims )
        self.information_masks = generate_information_masks( dims )
        self.position_layer = PositionVariance()
        
    def log_prob( self, samples ):
        i0 = self.position_layer( samples*self.information_masks[0] )
        l0 = self.parallelcnns[0]( i0 ).log_prob( samples )
        t0 = self.pixel_channel_groups[0]*l0

        i1 = self.position_layer( samples*self.information_masks[1] )
        l1 = self.parallelcnns[1]( i1 ).log_prob( samples )
        t1 = self.pixel_channel_groups[1]*l1

        i2 = self.position_layer( samples*self.information_masks[2] )
        l2 = self.parallelcnns[2]( i2 ).log_prob( samples )
        t2 = self.pixel_channel_groups[2]*l2
        
        i3 = self.position_layer( samples*self.information_masks[3] )
        l3 = self.parallelcnns[3]( i3 ).log_prob( samples )
        t3 = self.pixel_channel_groups[3]*l3

        tstack = tf.stack( [ t0, t1, t2, t3 ] )
        t = tf.reduce_sum( tstack, axis = 0 )
        return tf.reduce_sum( t, axis = [ 1, 2, 3 ] )
    
    def sample( self ):
        sample = tf.Variable( np.zeros( [ 1, 28, 28, 1 ] ).astype( np.float32 ) )
        
        i0 = self.position_layer( sample*self.information_masks[0] )
        l0 = tf.cast( self.parallelcnns[0]( i0 ).sample()*self.pixel_channel_groups[0], tf.float32 )
        sample = sample + l0
        
        i1 = self.position_layer( sample*self.information_masks[1] )
        l1 = tf.cast( self.parallelcnns[1]( i1 ).sample()*self.pixel_channel_groups[1], tf.float32 )
        sample = sample + l1

        i2 = self.position_layer( sample*self.information_masks[2] )
        l2 = tf.cast( self.parallelcnns[2]( i2 ).sample()*self.pixel_channel_groups[2], tf.float32 )
        sample = sample + l2
        
        i3 = self.position_layer( sample*self.information_masks[3] )
        l3 = tf.cast( self.parallelcnns[3]( i3 ).sample()*self.pixel_channel_groups[3], tf.float32 )
        sample = sample + l3
        
        return sample

