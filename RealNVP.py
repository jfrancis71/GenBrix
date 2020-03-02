
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

class StableScaleNet:
    def __init__( self, dims ):
        self.c1 = tf.Variable( np.zeros( dims ).astype( np.float32 ) )
        self.c2 = tf.Variable( np.zeros( dims ).astype( np.float32 ) )
        
    def forward( self, input ):
        return tf.tanh( input ) * self.c1 + self.c2
    
    def get_trainable_variables( self ):
        return [ self.c1 ] + [ self.c2 ]

def coupling_net( channels ):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=16, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.Conv2D(
            filters=channels, kernel_size=(3,3), padding='SAME' )
])

# PassThroughMask has 1 meaning it is a dependent variable, ie won't be changed on this coupling.
class CouplingLayer():
    def __init__( self, passThroughMask ):
        self.passThroughMask = passThroughMask
        self.cnet1 = coupling_net( passThroughMask.shape[2] )
        self.cnet2 = coupling_net( passThroughMask.shape[2] )
        self.stable1 = StableScaleNet( passThroughMask.shape )
        self.stable2 = StableScaleNet( passThroughMask.shape )
        
    def forward( self, input ):
        mu = self.stable1.forward( self.cnet1( self.passThroughMask*input ) )
        logscale = self.stable2.forward( self.cnet2( self.passThroughMask*input ) )
        changed = (input-mu)/tf.exp(logscale)
        transformed = self.passThroughMask*input + (1-self.passThroughMask)*changed
        jacobian = -logscale * ( 1 - self.passThroughMask )
        sum_jacobian = tf.reduce_sum( jacobian, axis = [ 1, 2, 3 ] )
        return [ transformed, sum_jacobian ]
        
    def reverse( self, input ):
        mu = self.stable1.forward( self.cnet1( self.passThroughMask*input ) )
        logscale = self.stable2.forward( self.cnet2( self.passThroughMask*input ) )
        changed = (input*tf.exp(logscale)) + mu
        result = self.passThroughMask*input + (1-self.passThroughMask)*changed
        return result
        
    def get_trainable_variables( self ):
        return self.cnet1.trainable_variables + self.cnet2.trainable_variables + \
            self.stable1.get_trainable_variables() + \
            self.stable2.get_trainable_variables()

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

class RealNVPBlock():
    def __init__( self, dims ):
        self.coupling_layer1 = CouplingLayer( checkerboardmask( dims ) )
        self.coupling_layer2 = CouplingLayer( 1-checkerboardmask( dims ) )
        self.squeeze_layer = SqueezeLayer()
        self.coupling_layer3 = CouplingLayer( channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ) )
        self.coupling_layer4 = CouplingLayer( 1-channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ) )
        
    def forward( self, input ):
        [ transformed1, jacobian1 ] = self.coupling_layer1.forward( input )
        [ transformed2, jacobian2 ] = self.coupling_layer2.forward( transformed1 )
        [ transformed3, jacobian3 ] = self.squeeze_layer.forward( transformed2 )
        [ transformed4, jacobian4 ] = self.coupling_layer3.forward( transformed3 )
        [ transformed5, jacobian5 ] = self.coupling_layer4.forward( transformed4 )
        return [ transformed5, jacobian1 + jacobian2 + jacobian3 + jacobian4 + jacobian5 ]

    def reverse( self, input ):
        transformed5 = self.coupling_layer4.reverse( input )
        transformed4 = self.coupling_layer3.reverse( transformed5 )
        transformed3 = self.squeeze_layer.reverse( transformed4 )
        transformed2 = self.coupling_layer2.reverse( transformed3 )
        transformed1 = self.coupling_layer1.reverse( transformed2 )
        return transformed1
    
    def get_trainable_variables( self ):
        return self.coupling_layer1.get_trainable_variables() + \
            self.coupling_layer2.get_trainable_variables() + \
            self.coupling_layer3.get_trainable_variables() + \
            self.coupling_layer4.get_trainable_variables()

class RealNVP(nb.Model):
    def __init__( self, dims ):
        super(RealNVP, self).__init__()
        self.dims = dims
        self.realnvp_block1 = RealNVPBlock( dims )
        self.realnvp_block2 = RealNVPBlock( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] )
        self.realnvp_block3 = RealNVPBlock( [ round(dims[0]/4), round(dims[1]/4), dims[2]*16 ] )
        self.coupling_layer1 = CouplingLayer( channelmask( [ round(dims[0]/8), round(dims[1]/8), dims[2]*64 ] ) )
        self.coupling_layer2 = CouplingLayer( 1-channelmask( [ round(dims[0]/8), round(dims[1]/8), dims[2]*64 ] ) )

    def forward( self, input ):
        [ transformed1, jacobian1 ] = self.realnvp_block1.forward( input )
        [ transformed2, jacobian2 ] = self.realnvp_block2.forward( transformed1 )
        [ transformed3, jacobian3 ] = self.realnvp_block3.forward( transformed2 )
        [ transformed4, jacobian4 ] = self.coupling_layer1.forward( transformed3 )
        [ transformed5, jacobian5 ] = self.coupling_layer2.forward( transformed4 )
        return [ transformed5, jacobian1 + jacobian2 + jacobian3 + jacobian4 + jacobian5 ]

    def loss( self, samples, logging_context=None, epoch=None ):
        [ transformed, jacobian ] = self.forward( samples )
        transformed_loss = -nb.log_normal_pdf( transformed, transformed*0.0, transformed*0.0)
        jacobian_loss = tf.reduce_mean( -jacobian )
        if logging_context is not None:
            tf.summary.scalar( logging_context+"_transformed_loss", transformed_loss, step=epoch )
            tf.summary.scalar( logging_context+"_jacobian_loss", jacobian_loss, step=epoch )
        return  transformed_loss + jacobian_loss

    def reverse( self, input ):
        transformed5 = self.coupling_layer2.reverse( input )
        transformed4 = self.coupling_layer1.reverse( transformed5 )
        transformed3 = self.realnvp_block3.reverse( transformed4 )
        transformed2 = self.realnvp_block2.reverse( transformed3 )
        transformed1 = self.realnvp_block1.reverse( transformed2 )
        return transformed1
    
    def sample( self, test_z=None ):
        z1 = np.random.normal( size = [ 1, round(self.dims[0]/8), round(self.dims[0]/8), self.dims[2]*64 ] ).astype( np.float32 )
        if test_z is None:
            test_z = z1
        return self.reverse( z1 )

    def get_trainable_variables( self ):
        return \
            self.realnvp_block1.get_trainable_variables() + \
            self.realnvp_block2.get_trainable_variables() + \
            self.realnvp_block3.get_trainable_variables() + \
            self.coupling_layer1.get_trainable_variables() + \
            self.coupling_layer2.get_trainable_variables()
