
import tensorflow as tf
import tensorflow_probability as tfp
from GenBrix import TFPVAEModels as vae_models

tfd = tfp.distributions
tfk = tf.keras

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

class VariationalAutoencoder:
    def __init__( self, vae_model, pixel_distribution=tfp.layers.IndependentBernoulli ):
        self.vae_model = vae_model
        self.encoder = self.vae_model.encoder()
        self.decoder = self.vae_model.decoder( pixel_distribution )
        self.vae = tfk.Model(inputs=self.encoder.inputs,
                outputs=self.decoder(self.encoder.outputs[0]))
        self.vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            loss=negloglik)
        
    def fit( self, examples, epochs=25 ):
        self.vae.fit( examples, examples, epochs=epochs )
        
    def log_prob( self, sample ):
        return negloglik( tf.expand_dims( sample, 0 ), self.vae( tf.expand_dims( sample, 0 ) ) )
    
    def sample( self ):
        return self.decoder( self.vae_model.prior.sample( 1 ) ).sample()[0]

# mymodel = vae.VariationalAutoencoder( vae_model=vae_models.YZVAEModel( latents = 16, q_distribution = tfp.layers.IndependentNormal ), pixel_distribution=tfp.layers.IndependentBernoulli )
# q_distribution could also be tfp.layers.MultivariateNormalTril
