
import numpy as np
import tensorflow as tf

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

class PixelCNN:
    def __init__( self, distribution, dims ):
        self.pixelcnns = create_pixelcnn_part( dims, distribution.no_of_parameters() )
        self.prediction_masks = generate_prediction_mask( dims )
        self.information_masks = generate_information_mask( dims )
        self.distribution = distribution
        self.glob_array = tf.Variable( np.zeros( [ dims[0], dims[1], 1 ] ).astype('float32') )
        self.dims = dims
        
    def loss( self, samples ):
        loss_dict={}
        sp = samples.shape
        for x in range( 4*sp[3] ):
            masked = samples * self.information_masks[x]
            sp = samples.shape
            broad_array = tf.broadcast_to( self.glob_array, [ sp[0], sp[1], sp[2], self.glob_array.shape[2] ] )
            info = tf.concat( [ masked, broad_array ],axis=3 )
            out1 = self.pixelcnns[x]( info )
            l = self.distribution.loss_per_prediction( out1, samples )
            loss_dict[x] = self.prediction_masks[x]*l
        
        lo = tf.stack( [ ot for ot in loss_dict.values() ])
        lo1 = tf.reduce_sum( lo, axis=0 )
        return tf.reduce_mean( tf.reduce_sum( lo1, axis = [ 1, 2, 3 ] ) )
    
    def sample( self ):
        shape = tf.shape( self.glob_array )
        image = np.zeros( [ 1, shape[0], shape[1], self.dims[2] ] )
        for l in range(4*self.dims[2]):
            broad_array = tf.broadcast_to( self.glob_array, [ image.shape[0], image.shape[1], image.shape[2] , 1 ] )
            info = tf.concat( [ image, broad_array ],axis=3 )
            image += self.distribution.sample(self.pixelcnns[l](info))*self.prediction_masks[l]
        return image

    def log_density( self, sample ):
        return -self.loss( tf.expand_dims( sample, 0 ) )
    
    def train( self, samples, no_epoch=10, learning_rate=.0001 ):
        train_dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(60000).batch(128)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        print( "Initial", "Training loss ", self.loss( samples[:128] ) )
        for epoch in range(no_epoch):
            for train_x in train_dataset:
                self.apply_gradients( optimizer, train_x)
            print( "Epoch", epoch, "Training loss ", self.loss( samples[:128] ) )
    
    def apply_gradients( self, optimizer, samples ):
        trainable_variables = flatten_lists( [ self.pixelcnns[r].trainable_variables for r in range(4) ] ) + [ self.glob_array ]
        with tf.GradientTape() as tape:
            xloss = self.loss( samples )
        g = tape.gradient( xloss, trainable_variables )
        optimizer.apply_gradients( zip ( g, trainable_variables ) )

