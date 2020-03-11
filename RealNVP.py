
#Achieves around -3,357 on 19,000 aligned CelebA faces trained for 30 epochs.

import numpy as np
import tensorflow as tf
from GenBrix import NBModel as nb

def checkerboardmask( shape ):
    zeros = np.zeros( shape ).astype( np.float32 )
    zeros[::2,::2,:] = 1.
    zeros[1::2,1::2,:] = 1.
    return zeros

def channelmask( shape ):
    zeros = np.zeros( shape ).astype( np.float32 )
    zeros[:,:,::2] = 1.
    return zeros

class StableScaleNet(tf.keras.layers.Layer):
    def __init__( self, dims ):
        super( StableScaleNet, self ).__init__()
        self.c1 = tf.Variable( np.zeros( dims ).astype( np.float32 ) )
        self.c2 = tf.Variable( np.zeros( dims ).astype( np.float32 ) )
        
    def call( self, input ):
        return tf.tanh( input ) * self.c1 + self.c2

def coupling_net( channels, mid_channels ):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=channels, kernel_size=(3,3), padding='SAME' )
])

# PassThroughMask has 1 meaning it is a dependent variable, ie won't be changed on this coupling.
class CouplingLayer(tf.keras.layers.Layer):
    def __init__( self, passThroughMask, mid_channels ):
        super( CouplingLayer, self ).__init__()
        self.passThroughMask = passThroughMask
        self.cnet1 = coupling_net( passThroughMask.shape[2], mid_channels )
        self.cnet2 = coupling_net( passThroughMask.shape[2], mid_channels )
        self.stable1 = StableScaleNet( passThroughMask.shape )
        self.stable2 = StableScaleNet( passThroughMask.shape )
        
    def call( self, input ):
        mu = self.stable1( self.cnet1( self.passThroughMask*input ) )
        logscale = self.stable2( self.cnet2( self.passThroughMask*input ) )
        changed = (input-mu)/tf.exp(logscale)
        transformed = self.passThroughMask*input + (1-self.passThroughMask)*changed
        jacobian = -logscale * ( 1 - self.passThroughMask )
        sum_jacobian = tf.reduce_sum( jacobian, axis = [ 1, 2, 3 ] )
        return [ transformed, sum_jacobian ]
        
    def reverse( self, input ):
        mu = self.stable1( self.cnet1( self.passThroughMask*input ) )
        logscale = self.stable2( self.cnet2( self.passThroughMask*input ) )
        changed = (input*tf.exp(logscale)) + mu
        result = self.passThroughMask*input + (1-self.passThroughMask)*changed
        return result

class SqueezeLayer():
    
    def forward( self, input ):
        shape = input.shape
        newx = tf.round( shape[2]/2 )
        newy = tf.round( shape[1]/2 )
        reshapedx = tf.reshape( input, [ shape[0],shape[1], newx, shape[3]*2 ])
        transposed = tf.transpose( reshapedx, [ 0, 2, 1, 3 ] )
        reshapedy = tf.reshape( transposed, [ shape[0], newx, newy, shape[3]*4 ])
        reshaped = tf.transpose( reshapedy, [ 0, 2, 1, 3 ])
        return [ reshaped, 0.0 ]
    
    def reverse( self, input ):
        shape = input.shape
        newx = tf.round( shape[2]*2 )
        newy = tf.round( shape[1]*2 )
        reshaped = tf.transpose( input, [ 0, 2, 1, 3 ])
        reshapedy = tf.reshape( reshaped, [ shape[0], shape[2], newy, round(shape[3]/2) ])
        transposed = tf.transpose( reshapedy, [ 0, 2, 1, 3 ] )
        reshapedx = tf.reshape( transposed, [ shape[0],newy, newx, round(shape[3]/4) ])
        return reshapedx

class RealNVPBlock( tf.keras.layers.Layer ):
    def __init__( self, dims, mid_channels ):
        super( RealNVPBlock, self ).__init__()
        self.coupling_layer1 = CouplingLayer( checkerboardmask( dims ), mid_channels )
        self.coupling_layer2 = CouplingLayer( 1-checkerboardmask( dims ), mid_channels )
        self.coupling_layer3 = CouplingLayer( checkerboardmask( dims ), mid_channels )
        self.squeeze_layer = SqueezeLayer()
        self.coupling_layer4 = CouplingLayer( channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ), mid_channels*2 )
        self.coupling_layer5 = CouplingLayer( 1-channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ), mid_channels*2 )
        self.coupling_layer6 = CouplingLayer( channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ), mid_channels*2 )
        
    def call( self, input ):
        [ transformed, jacobian1 ] = self.coupling_layer1( input )
        [ transformed, jacobian2 ] = self.coupling_layer2( transformed )
        [ transformed, jacobian3 ] = self.coupling_layer3( transformed )
        [ transformed, jacobian4 ] = self.squeeze_layer.forward( transformed )
        [ transformed, jacobian5 ] = self.coupling_layer4( transformed )
        [ transformed, jacobian6 ] = self.coupling_layer5( transformed )
        [ transformed, jacobian7 ] = self.coupling_layer6( transformed )
        return [ transformed, jacobian1 + jacobian2 + jacobian3 + jacobian4 + jacobian5 + jacobian6 + jacobian7 ]

    def reverse( self, input ):
        transformed = self.coupling_layer6.reverse( input )
        transformed = self.coupling_layer5.reverse( transformed )
        transformed = self.coupling_layer4.reverse( transformed )
        transformed = self.squeeze_layer.reverse( transformed )
        transformed = self.coupling_layer3.reverse( transformed )
        transformed = self.coupling_layer2.reverse( transformed )
        transformed = self.coupling_layer1.reverse( transformed )
        return transformed

class RealNVP(nb.Model):
    def __init__( self, dims ):
        super(RealNVP, self).__init__()
        self.dims = dims
        self.realnvp_block1 = RealNVPBlock( dims, 64 )
        self.realnvp_block2 = RealNVPBlock( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ], 128 )
#        self.realnvp_block3 = RealNVPBlock( [ round(dims[0]/4), round(dims[1]/4), dims[2]*16 ] )
        self.coupling_layer1 = CouplingLayer( 1-channelmask( [ round(dims[0]/4), round(dims[1]/4), dims[2]*16 ] ), 256 )

    def forward( self, input ):
        [ transformed, jacobian1 ] = self.realnvp_block1( input )
        [ transformed, jacobian2 ] = self.realnvp_block2( transformed )
#        [ transformed, jacobian3 ] = self.realnvp_block3( transformed )
        [ transformed, jacobian3 ] = self.coupling_layer1( transformed )
        return [ transformed, jacobian1 + jacobian2 + jacobian3 ]

    def loss( self, samples, logging_context=None, epoch=None ):
        [ transformed, jacobian ] = self.forward( samples )
        transformed_loss = -nb.log_normal_pdf( transformed, transformed*0.0, transformed*0.0)
        jacobian_loss = tf.reduce_mean( -jacobian )
        if logging_context is not None:
            tf.summary.scalar( logging_context+"_transformed_loss", transformed_loss, step=epoch )
            tf.summary.scalar( logging_context+"_jacobian_loss", jacobian_loss, step=epoch )
        return  transformed_loss + jacobian_loss

    def reverse( self, input ):
        transformed = self.coupling_layer1.reverse( input )
#        transformed = self.realnvp_block3.reverse( transformed4 )
        transformed = self.realnvp_block2.reverse( transformed )
        transformed = self.realnvp_block1.reverse( transformed )
        return transformed
    
    def sample( self, test_z=None ):
        z1 = np.random.normal( size = [ 1, round(self.dims[0]/4), round(self.dims[0]/4), self.dims[2]*16 ] ).astype( np.float32 )
        if test_z is None:
            test_z = z1
        return tf.math.sigmoid( self.reverse( z1 ) )

    def get_trainable_variables( self ):
        return \
            self.realnvp_block1.trainable_variables + \
            self.realnvp_block2.trainable_variables + \
            self.coupling_layer1.trainable_variables
