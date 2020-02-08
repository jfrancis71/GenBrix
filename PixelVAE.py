
from GenBrix import VariationalAutoencoder as vae
from GenBrix import NBModel as nb
from GenBrix import PixelCNN as cnn

class PixelVAE(nb.Model):
    def __init__( self, distribution, image_dims ):
        super(PixelVAE, self).__init__()
        self.vae = vae.VariationalAutoEncoder( cnn.ConditionalPixelCNN( distribution, image_dims ), image_dims )

    def loss( self, samples ):
        return self.vae.loss( samples )
    
    def sample( self, test_z=None ):
        return self.vae.sample( test_z )

    def apply_gradients( self, optimizer, samples ):
        self.vae.apply_gradients( optimizer, samples )