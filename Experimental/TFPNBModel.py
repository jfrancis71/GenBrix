import tensorflow as tf
import tensorflow_probability as tfp

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

BernoulliDistribution = ( 1 , lambda t: tfd.Independent(
                tfd.Bernoulli( logits=t[...,0] ),
                reinterpreted_batch_ndims=3 ) )

NormalDistribution = ( 2 , lambda t: tfd.Independent(
                tfd.Normal( loc=t[...,0], scale=.05 + tf.nn.softplus( t[...,1] ) ),
                reinterpreted_batch_ndims=3 ) )

class GBNBModel:
    def __init__( self, dims, distribution ):
        self.model = tf.keras.Sequential([
            tfp.layers.VariableLayer(shape=[ dims[0], dims[1], dims[2], distribution[0] ] ),
            tfp.layers.DistributionLambda( distribution[1] )
])
        self.model.compile( loss = negloglik, optimizer=tf.keras.optimizers.Adam(lr=0.03) )

    def fit( self, examples ):
        self.model.fit( examples, examples )

    def log_prob( self, sample ):
        return self.model(0).log_prob( sample )
    
    def sample( self ):
        return self.model(0).sample()

#mymodel = GBNBModel( [ 28, 28, 1 ], NormalDistribution )
#mymodel.fit( train_bin_images )
#mymodel.sample()[:,:,0]
