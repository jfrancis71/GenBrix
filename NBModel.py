
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
    
    def get_trainable_variables():
        return "unimplemented"
    
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
    
    def train( self, samples, no_epoch=10, learning_rate=.0001, log_dir=None ):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        print( "Initial", "Training loss ", self.loss( samples[:128] ) )
        if log_dir is not None:
            log_writer = tf.summary.create_file_writer(log_dir)
            with log_writer.as_default():
                self.train_loop( optimizer, samples, no_epoch, log_writer )
        else:
            self.train_loop( optimizer, samples, no_epoch, None)

#   Note, here we're just using the 1st batches of training and validation loss
#   as performance metrics
    def train_loop( self, optimizer, samples, no_epoch, log_writer ):
        randomized_samples = tf.random.shuffle( samples )
        train_size = np.round(randomized_samples.shape[0]*0.9).astype(int)
        train_set = randomized_samples[:train_size]
        validation_set = randomized_samples[train_size:min(train_size+128,randomized_samples.shape[0]-1)]
        for epoch in range(no_epoch):
            train_dataset = tf.data.Dataset.from_tensor_slices(train_set).shuffle(60000).batch(128)
            for train_x in train_dataset:
                self.apply_gradients( optimizer, train_x)
            train_loss_value = self.loss( train_set[:128])
            validation_loss_value = self.loss( validation_set )
            print( "Epoch", epoch, "Training loss ", train_loss_value, "Validation loss ",validation_loss_value )
            if log_writer is not None:
                tf.summary.scalar( 'training_loss', train_loss_value, step=epoch )
                tf.summary.scalar( 'validation_loss', validation_loss_value, step=epoch )
                samp = self.sample()
                tf.summary.image( 'sample', samp.astype(np.float32) , step=epoch )
                tf.summary.image( 'data sample', samples[:1], step=epoch)
    def sample():
        return "unimplemented"
    def loss( samples ):
        return "unimplemented"

    def apply_gradients( self, optimizer, samples ):
        trainable_variables = self.get_trainable_variables()
        with tf.GradientTape() as tape:
            xloss = self.loss( samples )
        g = tape.gradient( xloss, trainable_variables )
        optimizer.apply_gradients( zip ( g, trainable_variables ) )
        

class Binary(Distribution):

    def no_of_parameters( self ):
        return 1

    def loss( self, channel, samples ):
        assert( len( channel.shape ) == 4 and len( samples.shape ) == 4 )
        logits_parameters_output = reshape_channel_to_parameters( channel, 1 )
        loss = tf.nn.sigmoid_cross_entropy_with_logits( logits=logits_parameters_output[:,:,:,:,0], labels = samples )
#You want the sum across YxXxC so you can compare different losses, but average across batch.
        return tf.math.reduce_mean( tf.math.reduce_sum( loss, axis = [ 1, 2, 3 ] ) )

    def sample( self, channel ):
        channel_shape = tf.shape( channel )
        assert( len( channel.shape ) == 4 )
        logits_parameters_output = reshape_channel_to_parameters( channel, 1 )
        return np.random.binomial( 1, tf.math.sigmoid( logits_parameters_output[:,:,:,:,0] ) )

class RealGauss(Distribution):

    def no_of_parameters( self ):
        return 2

    def loss( self, channel, samples ):
        assert( len( channel.shape ) == 4 and len( samples.shape ) == 4 )
        logits_parameters_output = reshape_channel_to_parameters( channel, 2 )
        loss = -log_normal_pdf( samples, logits_parameters_output[:,:,:,:,0], logits_parameters_output[:,:,:,:,1] )
        return loss

    def sample( self, channel ):
        channel_shape = tf.shape( channel )
        assert( len( channel.shape ) == 4 )
        logits_parameters_output = reshape_channel_to_parameters( channel, 2 )
        logits_params_shape = logits_parameters_output.shape
        random_sample = tf.random.normal( shape = logits_params_shape[0:4] )
        return tf.exp( logits_parameters_output[:,:,:,:,1]/2 ) * random_sample + logits_parameters_output[:,:,:,:,0]

class Discrete(Distribution):

    def no_of_parameters( self ):
        return 10

    def loss( self, channel, samples ):
        assert( len( channel.shape ) == 4 and len( samples.shape ) == 4 )
        logits_parameters_output = reshape_channel_to_parameters( channel, 10 )
        scale_input = tf.multiply( samples, scale_const )
        rounds = tf.cast( tf.clip_by_value( tf.round( scale_input ), 0, 9 ), tf.int64 )
        cross = tf.nn.sparse_softmax_cross_entropy_with_logits( rounds, logits_parameters_output )
        loss = tf.math.reduce_mean( tf.math.reduce_sum( cross, axis = [ 1,2, 3 ] ) )
        return loss

    def sample( self, channel ):
        channel_shape = tf.shape( channel )
        assert( len( channel.shape ) == 4 )
        logit_parameters_output = reshape_channel_to_parameters( channel, 10 )
        parameters_output = tf.nn.softmax( logit_parameters_output )
        return np.apply_along_axis( sample, 4, parameters_output.numpy() )/10

#Reshapes channel so that it goes from b*y*x*(array channels) -> b*y*x*(input channels)*p
#where p is number of parameters of distribution
def reshape_channel_to_parameters( channel, no_parameters ):
    channel_shape = tf.shape( channel )
    no_image_channels = tf.math.floordiv( channel_shape[3], no_parameters )
    return tf.reshape ( channel, [ channel_shape[0], channel_shape[1], channel_shape[2], no_image_channels, no_parameters ] )

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

    def get_trainable_variables( self ):
        return [ self.array ]
    
scale_const = tf.constant( 10.0 )
