
import tensorflow as tf
import numpy as np

class Distribution:
    def no_of_parameters( self ):
        return "unimplemented"
# array and samples both of form (B,Y,X,C)
    def loss( self, array, samples ):
        return "unimplemented"
# array and samples both of form (B,Y,X,C)
    def sample( self, array1 ):
        return "unimplemented"
    
class Model:
    def log_density( self, sample ):
        return -self.loss( tf.expand_dims( sample, 0 ) )
    
    def log_densities( self, samples ):
        train_dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(60000).batch(64)
        mean = 0
        count = 0
        for train_x in train_dataset:
            mean += self.distribution.loss( self.array, train_x )
            count += 1
        return mean/count
    
    def train( self, samples, no_epoch=10, learning_rate=.0001, logging=False ):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        print( "Initial", "Training loss ", self.loss( samples[:128] ) )
        log_writer = None
        if logging is not None:
            log_writer = tf.summary.create_file_writer("./log")
            with log_writer.as_default():
                self.train_loop( optimizer, samples, no_epoch, log_writer )
        else:
            self.train_loop( optimizer, samples, no_epoch, None)
            
    def train_loop( self, optimizer, samples, no_epoch, log_writer ):
        train_dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(60000).batch(128)
        for epoch in range(no_epoch):
            for train_x in train_dataset:
                self.apply_gradients( optimizer, train_x)
            loss_value = self.loss( samples[:128])
            print( "Epoch", epoch, "Training loss ", loss_value )
            if log_writer is not None:
                tf.summary.scalar( 'loss', loss_value, step=epoch )
                samp = self.sample()
                tf.summary.image( 'sample', samp.astype(np.float32) , step=epoch )
                tf.summary.image( 'data sample', samples[:1], step=epoch)
    def sample():
        return "unimplemented"
    def loss( samples ):
        return "unimplemented"
    def apply_gradients():
        return "unimplemented"
        

class Binary(Distribution):
    def no_of_parameters( self ):
        return 1

    def loss_per_prediction( self, array, samples ):
        assert( len( array.shape ) == 4 and len( samples.shape ) == 4 )
        loss = tf.nn.sigmoid_cross_entropy_with_logits( logits=array, labels = samples )
        return loss

    def loss( self, array, samples ):
        assert( len( array.shape ) == 4 and len( samples.shape ) == 4 )
        loss = tf.nn.sigmoid_cross_entropy_with_logits( logits=array, labels = samples )
#You want the sum across YxXxC so you can compare different losses, but average across batch.
        return tf.math.reduce_mean( tf.math.reduce_sum( loss, axis = [ 1, 2, 3 ] ) )
    def sample( self, array ):
        array_shape = tf.shape( array )
        assert( len( array.shape ) == 4 )
        return np.random.binomial( 1, tf.math.sigmoid( array ) )

class RealGauss(Distribution):
    def no_of_parameters( self ):
        return 2

    def loss_per_prediction( self, array, samples ):
        assert( len( array.shape ) == 4 and len( samples.shape ) == 4 )
        reshape = reshape_array( array, samples, 2 )
        loss = -log_normal_pdf_per_prediction( samples, reshape[:,:,:,:,0], reshape[:,:,:,:,1] )
        return loss

    def loss( self, array, samples ):
        assert( len( array.shape ) == 4 and len( samples.shape ) == 4 )
        reshape = reshape_array( array, samples, 2 )
        loss = -log_normal_pdf( samples, reshape[:,:,:,:,0], reshape[:,:,:,:,1] )
        return loss
    def sample( self, array ):
        array_shape = tf.shape( array )
        assert( len( array.shape ) == 4 )
        no_image_channels = tf.math.floordiv( array_shape[3], 2 )
        reshape = tf.reshape ( array, [ array_shape[0], array_shape[1], array_shape[2], no_image_channels, 2 ] )
        return np.random.normal(
            reshape[:,:,:,:,0],
            np.sqrt( np.exp( reshape[:,:,:,:,1] ) ) )

class Discrete(Distribution):
    def no_of_parameters( self ):
        return 10

    def loss_per_prediction( self, array, samples ):
        assert( len( array.shape ) == 4 and len( samples.shape ) == 4 )
        broad_discrete = reshape_array( array, samples, 10 )
        scale_input = tf.multiply( samples, scale_const )
        rounds = tf.cast( tf.clip_by_value( tf.round( scale_input ), 0, 9 ), tf.int64 )
        cross = tf.nn.sparse_softmax_cross_entropy_with_logits( rounds, broad_discrete )
        loss = cross
        return loss

    def loss( self, array, samples ):
        assert( len( array.shape ) == 4 and len( samples.shape ) == 4 )
        broad_discrete = reshape_array( array, samples, 10 )
        scale_input = tf.multiply( samples, scale_const )
        rounds = tf.cast( tf.clip_by_value( tf.round( scale_input ), 0, 9 ), tf.int64 )
        cross = tf.nn.sparse_softmax_cross_entropy_with_logits( rounds, broad_discrete )
        loss = tf.math.reduce_mean( tf.math.reduce_sum( cross, axis = [ 1,2, 3 ] ) )
        return loss
    def sample( self, array ):
        array_shape = tf.shape( array )
        assert( len( array.shape ) == 4 )
        no_image_channels = tf.math.floordiv( array_shape[3], 10 )
        reshape = tf.reshape( array, [ array_shape[0], array_shape[1], array_shape[2], no_image_channels, 10 ] )
        soft = tf.nn.softmax( reshape )
        return np.apply_along_axis( sample, 4, soft.numpy() )/10

#Reshapes array so that it goes from b*y*x*(array channels) -> b*y*x*(input channels)*p
#where p is number of parameters of distribution
def reshape_array( array, samples, no_parameters ):
    samples_shape = tf.shape( samples )
    array_shape = tf.shape( array )
    return tf.reshape ( array, [ array_shape[0], array_shape[1], array_shape[2], samples_shape[3], no_parameters ] )

#returns scalar
def log_normal_pdf(samples, mean, logvar):
    assert( len( samples.shape ) == 4 and len( mean.shape ) == 4 and len( logvar.shape ) == 4 )
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_mean( tf.reduce_sum(
      -.5 * ((samples - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = [ 1, 2, 3 ]
      ) )

def log_normal_pdf_per_prediction(samples, mean, logvar):
    assert( len( samples.shape ) == 4 and len( mean.shape ) == 4 and len( logvar.shape ) == 4 )
    log2pi = tf.math.log(2. * np.pi)
    return -.5 * ((samples - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

def sample( probs ):
    return np.random.choice( 10, 1, p=probs )[0]

def broadcast_array( array, samples ):
    array_shape = array.shape
    samples_shape = samples.shape
    assert( len ( array_shape ) == 3 )
    assert( len ( samples_shape ) == 4 )
    return tf.broadcast_to( array, [ samples_shape[0], array_shape[0], array_shape[1], array_shape[2] ] )

class NBModel(Model):
    def __init__(self, distribution, dims):
        super(NBModel, self).__init__()
        self.distribution = distribution
        dims[2] = dims[2] * distribution.no_of_parameters()
        self.array = tf.Variable(
            np.zeros( dims ).astype('float32') )
    
    def loss( self, samples ):
        xbroadcast_array = broadcast_array( self.array, samples )
        return self.distribution.loss( xbroadcast_array, samples )

    def sample( self ):
        return self.distribution.sample( tf.expand_dims( self.array, 0 ) )

    def apply_gradients( self, optimizer, samples ):
        with tf.GradientTape() as tape:
            xbroadcast_array = broadcast_array( self.array, samples )
            loss = self.distribution.loss( xbroadcast_array, samples )
        gradients = tape.gradient( loss, self.array )
        optimizer.apply_gradients( zip( [ gradients ], [ self.array ] ) )

scale_const = tf.constant( 10.0 )
