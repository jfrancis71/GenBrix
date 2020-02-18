
from GenBrix import VariationalAutoencoder as vae
from GenBrix import NBModel as nb
from GenBrix import PixelCNN as cnn

class PixelVAE(nb.Model):
    def __init__( self, distribution, image_dims, vae_model=None ):
        super(PixelVAE, self).__init__()
        self.cnn = cnn.ConditionalPixelCNN( distribution, image_dims )
        if vae_model is None:
            self.vae = vae.VariationalAutoencoder( self.cnn, image_dims )
        else:
            self.vae = vae.VariationalAutoencoder( self.cnn, image_dims, vae_model )

    def loss( self, samples, logging_context=None, epoch=None ):
        return self.vae.loss( samples )
    
    def sample( self, test_z=None ):
        return self.vae.sample( test_z )

    def get_trainable_variables( self ):
        return self.vae.get_trainable_variables() + self.cnn.get_trainable_variables()
        
