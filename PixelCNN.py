
import numpy as np
import tensorflow as tf
from GenBrix import NBModel as nb

def generate_prediction_mask_2d( height, width ):#1 means predict on this iteration
    prediction_mask_2d = np.zeros( [ 4, height, width ] )
    prediction_mask_2d[0,::2,::2] = 1
    prediction_mask_2d[1,1::2,1::2] = 1
    prediction_mask_2d[2,::2,1::2] = 1
    prediction_mask_2d[3,1::2,0::2] = 1
    return prediction_mask_2d

def generate_prediction_mask( dims ):
    prediction_mask_build = np.zeros( [ 4, dims[2], dims[0], dims[1], dims[2] ])
    pred_mask_2d = generate_prediction_mask_2d( dims[0], dims[1] )
    for p in range(4):
        for ch in range(dims[2]):
            prediction_mask_build[p,ch,:,:,ch] = pred_mask_2d[p,:,:]
    prediction_mask = prediction_mask_build.reshape( [ dims[2]*4, dims[0], dims[1], dims[2] ] )
    return prediction_mask

#1 means you are allowed to see this, 0 means must be blocked
def generate_information_mask( dims ):
    prediction_mask = generate_prediction_mask( dims )
    information_mask = np.array( [ np.sum( prediction_mask[:x], axis=0 ) if x > 0 else np.zeros( [ dims[0], dims[1], dims[2] ] ) for x in range(4*dims[2]) ] )
    return information_mask

def create_pixelcnn_part( dims, distribution_parameters ):
    return [ tf.keras.Sequential([
    tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, padding='SAME',strides=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, padding='SAME',strides=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, padding='SAME',strides=(1, 1), activation='relu'),
    tf.keras.layers.Conv2D(
              filters=dims[2]*distribution_parameters, kernel_size=3, padding='SAME',strides=(1, 1), activation=None),
]) for x in range(4*dims[2]) ]

def flatten_lists( l ):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

class PixelCNN(nb.Model):
    def __init__( self, distribution, dims ):
        super(PixelCNN,self).__init__()
        self.distribution = ConditionalPixelCNN( distribution, dims )
        self.glob_array = tf.Variable( np.zeros( [ dims[0], dims[1], dims[2] ] ).astype('float32') )
        
    def loss( self, samples ):
        sp = samples.shape
        return self.distribution.loss( tf.broadcast_to( self.glob_array, [ sp[0], sp[1], sp[2], self.glob_array.shape[2] ] ), samples )
    
    def sample( self ):
        return self.distribution.sample( tf.expand_dims( self.glob_array, 0 ) )

    def get_trainable_variables( self ):
        return self.distribution.get_trainable_variables() + [ self.glob_array ]
    

from GenBrix import NBModel as nb

class ConditionalPixelCNN(nb.Distribution):

    def __init__( self, distribution, dims ):
        self.distribution = distribution
        self.pixelcnns = create_pixelcnn_part( dims, distribution.no_of_parameters() )
        self.prediction_masks = generate_prediction_mask( dims )
        self.information_masks = generate_information_mask( dims )

    def no_of_parameters( self ):
        return 1

    def loss_per_prediction( self, array, samples ):
        loss_dict={}
        sp = samples.shape
        for x in range( 4*sp[3] ):
            masked = samples * self.information_masks[x]
            sp = samples.shape
            info = tf.concat( [ masked, array ],axis=3 )
            out1 = self.pixelcnns[x]( info )
            l = self.distribution.loss_per_prediction( out1, samples )
            loss_dict[x] = self.prediction_masks[x]*l
        
        lo = tf.stack( [ ot for ot in loss_dict.values() ])
        lo1 = tf.reduce_sum( lo, axis=0 )
        return lo1


    def loss( self, array, samples ):
        return tf.reduce_mean( tf.reduce_sum( self.loss_per_prediction( array, samples ), axis = [ 1, 2, 3 ] ) )

    def sample( self, array ):
        shape = tf.shape( array )
        assert( len( array.shape ) == 4 )
        no_image_channels = tf.math.floordiv( shape[3], 1 )
        image = np.zeros( [ 1, shape[1], shape[2], no_image_channels ] )
        for l in range(4*no_image_channels):
            info = tf.concat( [ image, array ],axis=3 )
            image += self.distribution.sample(self.pixelcnns[l](info))*self.prediction_masks[l]
        return image
    
    def get_trainable_variables( self ):
        return flatten_lists( [ self.pixelcnns[r].trainable_variables for r in range(4) ] )
    
