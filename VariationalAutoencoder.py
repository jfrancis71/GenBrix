import tensorflow as tf
import numpy as np

from GenBrix import NBModel as nb

#This is a convolution latent variable version of Tensorflow demo example
def inference_net():
    return tf.keras.Sequential([
    tf.keras.layers.Conv2D(
              filters=32, kernel_size=3, padding='SAME',strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(
              filters=64, kernel_size=3, padding='SAME',strides=(2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense( units=50*2, activation=None),
    tf.keras.layers.Reshape( target_shape=(1,1,50,2))
])

#This is a convolution latent variable version of Tensorflow demo example
#I build the parameters for a probability distribution over an image given
#some latent variables, therefore I need to know the size of the image.
def generative_net( image_dims, no_distribution_parameters ):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
              filters=(image_dims[0]//4)*(image_dims[1]//4)*32, kernel_size=1, padding='SAME',strides=(1, 1), activation='relu'),
        tf.keras.layers.Reshape( target_shape=(image_dims[0]//4,image_dims[1]//4,32) ),
        tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
        tf.keras.layers.Conv2DTranspose(
        filters=image_dims[2]*no_distribution_parameters, kernel_size=3, strides=(1, 1), padding="SAME", activation=None)
    ])

def sample_latent( z1 ):
    mean_sample = z1[:,:,:,:,0]
    logvar_sample = z1[:,:,:,:,1]
    random_sample = tf.random.normal( shape = logvar_sample.shape )
    return tf.exp( logvar_sample ) * random_sample + mean_sample

class VariationalAutoEncoder:
    def __init__( self, distribution, image_dims ):
        self.xinference_net = inference_net()
        self.xgenerative_net = generative_net( image_dims, distribution.no_of_parameters() )
        self.distribution = distribution

    def loss( self, samples ):
        zm = self.xinference_net( samples )
        z1 = sample_latent( zm )
        de = self.xgenerative_net( z1  )
        l1 = self.distribution.loss( de, samples )
        logpz = nb.log_normal_pdf(z1, zm[:,:,:,:,0]*0.0, zm[:,:,:,:,0]*0. )
        logqz_x = nb.log_normal_pdf(z1, zm[:,:,:,:,0], zm[:,:,:,:,1] )
        kl_loss = logqz_x - logpz
        loss = tf.reduce_mean( l1 + kl_loss )
        return loss
    
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
        
    def sample( self, test_z ):
        return self.distribution.sample( self.xgenerative_net( test_z ) )

    def train( self, samples, no_epoch=10, learning_rate=.0001 ):
        train_dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(60000).batch(128)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        print( "Initial", "Training loss ", self.loss( samples ) )
        for epoch in range(no_epoch):
            for train_x in train_dataset:
                self.apply_gradients( optimizer, train_x)
            print( "Epoch", epoch, "Training loss ", self.loss( samples[:128] ) )
    
    def apply_gradients( self, optimizer, samples ):
        trainable_variables = self.xinference_net.trainable_variables + self.xgenerative_net.trainable_variables
        with tf.GradientTape() as tape:
            xloss = self.loss( samples )
        g = tape.gradient( xloss, trainable_variables )
        optimizer.apply_gradients( zip ( g, trainable_variables ) )


