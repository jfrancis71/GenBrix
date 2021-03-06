
#Achieves around -3,460 on 19,000 aligned CelebA faces trained for 30 epochs.
#albeit -3,691 after 43 epochs. (All for batch size 64)

import numpy as np
import tensorflow as tf
from GenBrix import NBModel as nb
import tensorflow_addons as tfa

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
    def __init__( self, channels ):
        super( StableScaleNet, self ).__init__()
        self.c1 = tf.Variable( np.zeros( channels ).astype( np.float32 ) )
        
    def call( self, input ):
        return tf.tanh( input ) * self.c1

def coupling_net( channels, mid_channels ):
    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(
            filters=channels, kernel_size=(3,3), padding='SAME' )
])

class ResidualBlock( tf.keras.layers.Layer ):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = tf.keras.layers.BatchNormalization()
        self.in_conv = tfa.layers.WeightNormalization( tf.keras.layers.Conv2D( out_channels, kernel_size=(3,3), padding='SAME', use_bias=False ) )

        self.out_norm = tf.keras.layers.BatchNormalization()
        self.out_conv = tfa.layers.WeightNormalization( tf.keras.layers.Conv2D( out_channels, kernel_size=(3,3), padding='SAME', use_bias=True ) )


    def call(self, x):
        skip = x

        x = self.in_norm(x)
        x = tf.nn.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = tf.nn.relu(x)
        x = self.out_conv(x)

        x = x + skip

        return x

class CouplingNet( tf.keras.layers.Layer ):
    def __init__( self, channels, mid_channels ):
        super( CouplingNet, self ).__init__()

        self.b1 = tf.keras.layers.BatchNormalization()
        self.c1 = tf.keras.layers.Conv2D(
            filters=mid_channels, kernel_size=(3,3), padding='SAME', activation='tanh' )

        self.blocks = [ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(4)]
        self.skips = [tfa.layers.WeightNormalization( tf.keras.layers.Conv2D(mid_channels, kernel_size=(1,1), padding='SAME' ) )
                                    for _ in range(4)]

        self.b2 = tf.keras.layers.BatchNormalization()
        self.c2 = tf.keras.layers.Conv2D(
            filters=channels, kernel_size=(1,1), padding='SAME', activation='tanh' )

    def call( self, input ):
        x = self.b1( input )
        x = self.c1( x )

        x_skip = x

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)

        x = self.b2( x_skip )
        x = self.c2( x )

        return x


# PassThroughMask has 1 meaning it is a dependent variable, ie won't be changed on this coupling.
class CouplingLayer(tf.keras.layers.Layer):
    def __init__( self, passThroughMask, mid_channels ):
        super( CouplingLayer, self ).__init__()
        self.passThroughMask = passThroughMask
        self.cnet1 = CouplingNet( passThroughMask.shape[2]*2, mid_channels )
#Intentionally keep in StableScaleNet for mu, it seems to stabilise learning...???
        self.stable1 = StableScaleNet( passThroughMask.shape[2] )
        self.stable2 = StableScaleNet( passThroughMask.shape[2] )
        
    def call( self, input ):
        cn = self.cnet1( self.passThroughMask*input )
        mu1, logvar1 = tf.split( cn, num_or_size_splits=2, axis=3 )
        mu = self.stable1( mu1 )
        logscale = self.stable2( logvar1 )
        changed = (input-mu)/tf.exp(logscale)
        transformed = self.passThroughMask*input + (1-self.passThroughMask)*changed
        jacobian = -logscale * ( 1 - self.passThroughMask )
        sum_jacobian = tf.reduce_sum( jacobian, axis = [ 1, 2, 3 ] )
        return [ transformed, sum_jacobian ]
        
    def reverse( self, input ):
        cn = self.cnet1( self.passThroughMask*input )
        mu1, logvar1 = tf.split( cn, num_or_size_splits=2, axis=3 )
        mu = self.stable1( mu1 )
        logscale = self.stable2( logvar1 )
        changed = (input*tf.exp(logscale)) + mu
        result = self.passThroughMask*input + (1-self.passThroughMask)*changed
        return result

class ChannelCouplingLayer( tf.keras.layers.Layer ):
    def __init__( self, identity, output_channels, mid_channels ):
        super( ChannelCouplingLayer, self ).__init__()
        self.identity = identity
        self.cnet1 = CouplingNet( output_channels, mid_channels )
        self.stable1 = StableScaleNet( output_channels//2 )
        self.stable2 = StableScaleNet( output_channels//2 )

    def call( self, input ):
        [ left, right ] = tf.split( input, num_or_size_splits=2, axis=3 )
        cn = self.cnet1( left if ( self.identity == 1 ) else right )
        mu1, logvar1 = tf.split( cn, num_or_size_splits=2, axis=3 )
        mu = self.stable1( mu1 )
        logscale = self.stable2( logvar1 )
        changed = ( (right if ( self.identity == 1 ) else left ) -mu)/tf.exp(logscale)
        transformed = tf.concat( [ left, changed ] if ( self.identity == 1 ) else [ changed, right ], axis = 3 )
        sum_jacobian = tf.reduce_sum( -logscale, axis = [ 1, 2, 3 ] )
        return [ transformed, sum_jacobian ]

    def reverse( self, input ):
        [ left, right ] = tf.split( input, num_or_size_splits=2, axis=3 )
        cn = self.cnet1( left if ( self.identity == 1 ) else right )
        mu1, logvar1 = tf.split( cn, num_or_size_splits=2, axis=3 )
        mu = self.stable1( mu1 )
        logscale = self.stable2( logvar1 )
        changed = ( (right if ( self.identity == 1 ) else left ) )*tf.exp(logscale) + mu
        transformed = tf.concat( [ left, mu ] if ( self.identity == 1 ) else [ mu, right ], axis = 3 )
        return transformed



class SqueezeLayer():
    
    def forward( self, input, alt_order=False ):
        shape = input.shape
        if ( alt_order == False ):
            tft1 = tf.reshape( input, [ shape[0], shape[1]//2, 2, shape[2]//2, 2, shape[3] ] )
            tft2 = tf.transpose( tft1, [ 0, 1, 3, 5, 2, 4 ] )
            tft3 = tf.reshape( tft2, [ shape[0], shape[1]//2, shape[2]//2, shape[3]*4 ])
        else:
            spl = tf.reshape( input, [ shape[0], shape[1]//2, 2, shape[2]//2, 2, shape[3]])
            tp = tf.transpose( spl, [ 0, 2, 4, 1, 3, 5])
            tp1 = tf.reshape( tp, [ shape[0], 4, shape[1]//2, shape[2]//2, shape[3] ])
            spl1 = tf.unstack( tp1, axis = 1 )
            tft3 = tf.concat( [ spl1[0], spl1[3], spl1[1], spl1[2] ], axis=3 )
        return [ tft3, 0.0 ]

    def reverse( self, tft3, alt_order=False ):
        shape = tft3.shape
        if ( alt_order == False ):
            tft2 = tf.reshape( tft3, [ shape[0], shape[1], shape[2], shape[3]//4, 2, 2 ] )
            tft1 = tf.transpose( tft2, [ 0, 1, 4, 2, 5, 3 ] )
            input = tf.reshape( tft1, [ shape[0], shape[1]*2, shape[2]*2, shape[3]//4 ] )
        else:
            spl1 = tf.split( tft3, num_or_size_splits=4, axis=3 )
            tp1 = tf.stack( [ spl1[0], spl1[2], spl1[3], spl1[1] ], axis=1 )
            tp = tf.reshape( tp1,  [ shape[0], 2, 2, shape[1], shape[2], shape[3]//4 ] )
            spl = tf.transpose( tp, [ 0, 3, 1, 4, 2, 5 ] )
            input = tf.reshape( spl, [ shape[0], shape[1]*2, shape[2]*2, shape[3]//4 ] )

        return input

class RealNVPBlock( tf.keras.layers.Layer ):
    def __init__( self, dims, mid_channels ):
        super( RealNVPBlock, self ).__init__()
        self.coupling_layer1 = CouplingLayer( checkerboardmask( dims ), mid_channels )
        self.coupling_layer2 = CouplingLayer( 1-checkerboardmask( dims ), mid_channels )
        self.coupling_layer3 = CouplingLayer( checkerboardmask( dims ), mid_channels )
        self.squeeze_layer = SqueezeLayer()
#        self.coupling_layer4 = CouplingLayer( channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ), mid_channels*2 )
#        self.coupling_layer5 = CouplingLayer( 1-channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ), mid_channels*2 )
#        self.coupling_layer6 = CouplingLayer( channelmask( [ round(dims[0]/2), round(dims[1]/2), dims[2]*4 ] ), mid_channels*2 )
        self.coupling_layer4 = ChannelCouplingLayer( 1, dims[2]*4, mid_channels*2 )
        self.coupling_layer5 = ChannelCouplingLayer( 2, dims[2]*4, mid_channels*2 )
        self.coupling_layer6 = ChannelCouplingLayer( 1, dims[2]*4, mid_channels*2 )

#Note should really be doing an unsqueeze followed by a squeeze with alt order at end 
    def call( self, input ):
        [ transformed, jacobian1 ] = self.coupling_layer1( input )
        [ transformed, jacobian2 ] = self.coupling_layer2( transformed )
        [ transformed, jacobian3 ] = self.coupling_layer3( transformed )
        [ transformed, jacobian4 ] = self.squeeze_layer.forward( transformed )
        [ transformed, jacobian5 ] = self.coupling_layer4( transformed )
        [ transformed, jacobian6 ] = self.coupling_layer5( transformed )
        [ transformed, jacobian7 ] = self.coupling_layer6( transformed )
        transformed = self.squeeze_layer.reverse( transformed )
        [ transformed, _ ] = self.squeeze_layer.forward( transformed, alt_order=True )
        return [ transformed, jacobian1 + jacobian2 + jacobian3 + jacobian4 + jacobian5 + jacobian6 + jacobian7 ]

    def reverse( self, input ):
        transformed = self.squeeze_layer.reverse( input, alt_order=True )
        [ transformed, _ ] = self.squeeze_layer.forward( transformed )
        transformed = self.coupling_layer6.reverse( transformed )
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
        self.coupling_layer1 = CouplingLayer( checkerboardmask( [ dims[0]//2, dims[1]//2, dims[2]*2 ] ), 128 )
        self.coupling_layer2 = CouplingLayer( 1-checkerboardmask( [ dims[0]//2, dims[1]//2, dims[2]*2 ] ), 128 )
        self.coupling_layer3 = CouplingLayer( checkerboardmask( [ dims[0]//2, dims[1]//2, dims[2]*2 ] ), 128 )
        self.coupling_layer4 = CouplingLayer( 1-checkerboardmask( [ dims[0]//2, dims[1]//2, dims[2]*2 ] ), 128 )

    def forward( self, input ):
        [ transformed, jacobian1 ] = self.realnvp_block1( input )
        [ left, right ] = tf.split( transformed, num_or_size_splits=2, axis=3 )
        [ transformed, jacobian2 ] = self.coupling_layer1( right )
        [ transformed, jacobian3 ] = self.coupling_layer2( transformed )
        [ transformed, jacobian4 ] = self.coupling_layer3( transformed )
        [ transformed, jacobian5 ] = self.coupling_layer4( transformed )
        transformed = tf.concat( [ left, transformed ], axis=3 )
        return [ transformed, jacobian1 + jacobian2 + jacobian3 + jacobian4 + jacobian5 ]

    def loss( self, samples, logging_context=None, epoch=None ):
        [ transformed, jacobian ] = self.forward( samples )
        transformed_loss = -nb.log_normal_pdf( transformed, transformed*0.0, transformed*0.0)
        jacobian_loss = tf.reduce_mean( -jacobian )
        if logging_context is not None:
            tf.summary.scalar( logging_context+"_transformed_loss", transformed_loss, step=epoch )
            tf.summary.scalar( logging_context+"_jacobian_loss", jacobian_loss, step=epoch )
        return  transformed_loss + jacobian_loss

    def reverse( self, input ):
        [ left, transformed ] = tf.split( input, num_or_size_splits=2, axis = 3 )
        transformed = self.coupling_layer4.reverse( transformed )
        transformed = self.coupling_layer3.reverse( transformed )
        transformed = self.coupling_layer2.reverse( transformed )
        transformed = self.coupling_layer1.reverse( transformed )
        transformed = tf.concat( [ left, transformed ], axis=3 )
        transformed = self.realnvp_block1.reverse( transformed )
        return transformed
    
    def sample( self, test_z=None ):
        z1 = np.random.normal( size = [ 1, round(self.dims[0]/2), round(self.dims[0]/2), self.dims[2]*4 ] ).astype( np.float32 )
        if test_z is None:
            test_z = z1
        return tf.math.sigmoid( self.reverse( z1 ) )

    def get_trainable_variables( self ):
        return \
            self.realnvp_block1.trainable_variables + \
            self.coupling_layer1.trainable_variables + \
            self.coupling_layer2.trainable_variables + \
            self.coupling_layer3.trainable_variables + \
            self.coupling_layer4.trainable_variables
