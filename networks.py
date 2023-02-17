"""
Creates an abstract model for a Variational Autoencoder
"""

import tensorflow.keras as keras
from keras import backend, layers

class Downsampling(layers.Layer):
      """
  Creates a convolution/deconvolution layer for a convolutional network
  that has a Convolution2D/Conv2DTranspose Layer, Batch normalisation and an activation'
  """
  def __init__(self, filters, kernel, strides=2, padding='same', 
               normalization_momentum:int=0.8, leaky_alpha=0.2, **kwargs):
    super().__init__(**kwargs)
    self.convolution = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)
    self.normalisation = None if normalization_momentum is None else layers.BatchNormalization(momentum=normalization_momentum)
    self.activation = layers.LeakyReLU(alpha=leaky_alpha)

  def call(self, inputs):
    x = self.convolution(inputs)
    if self.normalisation:
      x = self.normalisation(x)
    x = self.activation(x)
    return x

  def summarize(self, input_sz):
    x = layers.Input(shape=input_sz, name='input')
    Model(x, self.call(x), name=self.name).summary()

class Upsampling(layers.Layer):
  """
  Creates a convolution/deconvolution layer for a convolutional network
  that has a Convolution2D/Conv2DTranspose Layer, Batch normalisation and an activation'
  """
  def __init__(self, filters, kernel=5, strides=2, padding='same', 
               normalization_momentum:int=0.8, **kwargs):
    super().__init__(**kwargs)
    self.convolution = layers.Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding=padding, activation='relu')
    self.normalisation = None if momentum is None else layers.BatchNormalization(momentum=normalization_momentum)

  def call(self, inputs):
    x = self.convolution(inputs)
    if self.normalisation:
      x = self.normalisation(x)
    return x

  def summarize(self, input_sz):
    x = layers.Input(shape=input_sz, name='input')
    Model(x, self.call(x), name=self.name).summary()



class Sampling(layers.Layer):
    """ Sampling Layer for a Variational Encoder
    The layer takes a mean tensor and a variance tensor to calculate a latent distribution
    """
    def call(self, inputs):
        distribution_mean, distribution_variance = inputs
        batch_size = backend.shape(distribution_variance)[0]
        random = backend.random_normal(shape=(batch_size, backend.shape(distribution_variance)[1]))
        latent_samples = distribution_mean + backend.exp(0.5 * distribution_variance) * random
        return latent_samples

class VariationalEncoder(layers.Layer):
    """
    An encoding layer for the Variational AE
    """

    def __init__(self, subnet, intermediate_dim=(28,28,12), latent_dim=32, flattened=False, **kwargs):
        super().__init__(name=subnet.name, **kwargs)
        self.subnet = subnet
        self.dense_projection = layers.Dense(np.product(intermediate_dim))
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        self.flatten = flattened

    def call(self, inputs):
        x = self.subnet(inputs)
        if flatten:
            x = layers.Flatten()(x)
        distribution_mean = self.dense_mean(x)
        distribution_variance = self.dense_log_var(x)
        latent_samples = self.sampling( (distribution_mean, distribution_variance))
        return latent_samples, distribution_mean, distribution_variance,


class VariationalDecoder(layers.Layer):
    """
    The decoding layer of a VAE, which takes a tensor of latent samples 
    and decode a separate object.
    """

    def __init__(self, subnet, intermediate_dim=(28, 28, 12), latent_dim=32, reshape=False, **kwargs):
        super().__init__(name=subnet.name, **kwargs) #, input_shape=latent_dim)
        self.expand = layers.Dense(np.product(intermediate_dim), input_shape=latent_dim)
        self.reshape = layers.Reshape(intermediate_dim) if reshape else None
        self.subnet = subnet

    def call(self, inputs):
        x = self.expand_dense(inputs)
        if self.reshape:
            x = self.reshape(x)
        decoded = self.subnet(x)
        return decoded

class VariationalAutoencoder(keras.Model):

    def __init__(self, encoder_sub, decoder_sub, latent_dim=32, intermediate_dim=(28, 28, 12), flatten=False):
        """
        Params:
            - encoder_sub           :
            - decoder_sub           :
            - latent_dim            :
            - intermediate_dim      : the size of the last layer of the encoder subnetwork
            - flatten               : whether to flatten the output of the encoder subnetwork. If `True`, the decoder subnetwork will be expanded as well.
            - input_size            : the typical size of the input
        """
        super().__init__(**kwargs)
        
        # The outer layers
        self.encoder = VariationalEncoder(encoder_sub, intermediate_dim=intermediate_dim,
            latent_dim = latent_dim, flatten=flatten)
        self.decoder = VariationalEncoder(decoder_sub, intermediate_dim=intermediate_dim,
            latent_dim = latent_dim, reshape=flatten)

    def call(self, inputs):
        latent_samples, latent_mean, latent_variance = self.encoder(inputs)
        outputs = self.decoder(latent_samples)

        # Add KL divergence regularization loss
        kl_loss = -0.5 * tf.reduce_mean(
            latent_variance - tf.square(latent_mean) - tf.exp(latent_variance) + 1
        )
        self. add_loss(kl_loss)

        return outputs


        


