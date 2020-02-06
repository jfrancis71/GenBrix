
from GenBrix import VariationalAutoencoder as vae
from GenBrix import NBModel as nb
from GenBrix import PixelCNN as cnn

class PixelVAE:
    def __init__( self, distribution, image_dims ):
        self.vae = vae.VariationalAutoEncoder( cnn.ConditionalPixelCNN( distribution, image_dims ), image_dims )

    def loss( self, samples ):
        return self.vae.loss( samples )

    def log_density( self, sample ):
        return self.vae.log_density( sample )

    def log_densities( self, samples ):
        return self.vae.log_densities( samples )

    def sample( self, test_z ):
        return self.vae.sample( test_z )

    def train( self, samples, no_epoch=10, learning_rate=.0001 ):
        self.vae.train( samples, no_epoch, learning_rate )
