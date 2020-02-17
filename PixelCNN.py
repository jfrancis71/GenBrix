
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
              filters=16, kernel_size=3, padding='SAME',strides=(1, 1), activation='tanh'),
    tf.keras.layers.Conv2D(
              filters=16, kernel_size=1, padding='SAME',strides=(1, 1), activation='tanh'),
    tf.keras.layers.Conv2D(
              filters=dims[2]*distribution_parameters, kernel_size=1, padding='SAME',strides=(1, 1), activation=None),
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
        self.conditional_array = tf.Variable( np.zeros( [ dims[0], dims[1], dims[2] ] ).astype('float32') )
        
    def loss( self, samples, logging_context=None, epoch=None ):
        sp = samples.shape
        return self.distribution.loss( tf.broadcast_to( self.conditional_array, [ sp[0], sp[1], sp[2], self.conditional_array.shape[2] ] ), samples )
    
    def sample( self ):
        return self.distribution.sample( tf.expand_dims( self.conditional_array, 0 ) )

    def get_trainable_variables( self ):
        return self.distribution.get_trainable_variables() + [ self.conditional_array ]
    

from GenBrix import NBModel as nb

class ConditionalPixelCNN(nb.Distribution):

    def __init__( self, distribution, dims ):
        self.distribution = distribution
        self.pixelcnns = create_pixelcnn_part( dims, distribution.no_of_parameters() )
        self.prediction_masks = generate_prediction_mask( dims )
        self.information_masks = generate_information_mask( dims )
        self.prediction_output_masks = np.zeros( [ 4*dims[2], dims[0], dims[1], dims[2]*distribution.no_of_parameters() ])
        for x in range( 4 * dims[2] ):
            prediction_parameter_masks = np.zeros( [ dims[0], dims[1], dims[2], distribution.no_of_parameters() ])
            for p in range( distribution.no_of_parameters() ):
                prediction_parameter_masks[:,:,:,p] = self.prediction_masks[x]
            self.prediction_output_masks[x] = np.reshape( prediction_parameter_masks, [ dims[0], dims[1], dims[2]*self.distribution.no_of_parameters() ] )

    def no_of_parameters( self ):
        return 1 

# It predicts distribution parameters, but in an autoregressive fashion, ie in terms of predicting samples, some of the
# info from sample is used.
    def predict_parameters( self, conditional, samples ):
        sp_shape = samples.shape
        conditional_net_masked_output_dict = {}
        for x in range( 4 * sp_shape[3] ):
            masked = samples * self.information_masks[x]
            conditional_net_input = tf.concat( [ masked, conditional ],axis=3 )
            conditional_net_output = self.pixelcnns[x]( conditional_net_input )
#            prediction_parameter_masks = np.zeros( [ sp_shape[1], sp_shape[2], sp_shape[3], self.distribution.no_of_parameters() ])
#            for p in range( self.distribution.no_of_parameters() ):
#                prediction_parameter_masks[:,:,:,p] = self.prediction_masks[x]
#            prediction_output_masks = np.reshape( prediction_parameter_masks, [ sp_shape[1], sp_shape[2], sp_shape[3]*self.distribution.no_of_parameters() ] )
            conditional_net_masked_output = self.prediction_output_masks[x] * conditional_net_output
            conditional_net_masked_output_dict[x] = conditional_net_masked_output
        conditional_net_masked_output_stack = tf.stack( [ conditional_net_masked_output for conditional_net_masked_output in conditional_net_masked_output_dict.values() ])
        conditional_net_output = tf.reduce_sum( conditional_net_masked_output_stack, axis=0 )
        return conditional_net_output

    def loss( self, conditional, samples ):
        params = self.predict_parameters( conditional, samples )
        return self.distribution.loss( params, samples )

    def sample( self, conditional ):
        shape = tf.shape( conditional )
        assert( len( conditional.shape ) == 4 )
        no_image_channels = tf.math.floordiv( shape[3], self.no_of_parameters() )
        image = np.zeros( [ 1, shape[1], shape[2], no_image_channels ] )
        for l in range(4*no_image_channels):
            info = tf.concat( [ image, conditional ],axis=3 )
            image += self.distribution.sample(self.pixelcnns[l](info))*self.prediction_masks[l]
        return image

    def get_trainable_variables( self ):
        return flatten_lists( [ cond.trainable_variables for cond in self.pixelcnns ] )
