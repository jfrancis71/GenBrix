
import tensorflow as tf
import tensorflow_probability as tfp
from GenBrix import TFPVAEModels as vae_models

tfd = tfp.distributions
tfk = tf.keras

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

BernoulliDistribution = ( 1 , lambda t: tfd.Independent(
                tfd.Bernoulli( logits=t[...,0] ),
                reinterpreted_batch_ndims=3 ) )

NormalDistribution = ( 2 , lambda t: tfd.Independent(
                tfd.Normal( loc=t[...,0], scale=.05 + tf.nn.softplus( t[...,1] ) ),
                reinterpreted_batch_ndims=3 ) )

class VariationalAutoencoder:
    def __init__( self, distribution=BernoulliDistribution, vae_model=None ):
        self.vae_model = vae_model
        self.encoder = self.vae_model.encoder()
        self.decoder = self.vae_model.decoder( distribution )
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
