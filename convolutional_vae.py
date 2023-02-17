"""
Deep Learning Models
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from keras import backend

class Sampling(layers.Layer):
  """ Sampling Layer for a convolutional VAE"""
  def call(self, inputs):
    distribution_mean, distribution_variance = inputs
    batch_size = backend.shape(distribution_variance)[0]
    random = backend.random_normal(shape=(batch_size, backend.shape(distribution_variance)[1]))
    latent_samples = distribution_mean + backend.exp(0.5 * distribution_variance) * random
    return latent_samples

class VariationalEncoder(layers.Layer):
  """
  Builds a convolutional VAE
  """
  def __init__(self, encoding_seq, latent_dim=28, intermediate_dim=(32,32,64),name='encoder', **kwargs):
    super().__init__(name=name,**kwargs)
    self.encoding_seq = encoding_seq
    self.dense_mean = layers.Dense(latent_dim)
    self.dense_log_var = layers.Dense(latent_dim)
    self.sampling = Sampling()
    self.intermediate_dim = intermediate_dim # Size of 
  
  def call(self, inputs):
    x = self.encoding_seq(inputs)  
    x = layers.Flatten()(x)
    distribution_mean = self.dense_mean(x)
    distribution_variance = self.dense_log_var(x)
    latent_samples = self.sampling((distribution_mean, distribution_variance))
    return latent_samples, distribution_mean, distribution_variance,


class VariationalDecoder(layers.Layer):
  def __init__(self, decoding_seq, intermediate_dim=(32,32,64), name='decoder', **kwargs):
    super().__init__(name=name,**kwargs)
    self.decoding_seq = decoding_seq
    self.intermediate_dim = intermediate_dim # Size of 
    self.expand_dense = layers.Dense(units=np.prod(self.intermediate_dim), activation="relu", name='expand')
    self.reshape = layers.Reshape(self.intermediate_dim, name='reshape')

  @tf.function
  def call(self, inputs):
    # Expand and reshape
    x = self.expand_dense(inputs)
    x = self.reshape(x)
    x = self.decoding_seq(x)
    return x

class VariationalAE(keras.Model):
  
  def __init__(self, encoding_seq, decoding_seq,
               input_size, intermediate_dim=(32,32,64),
               latent_dim=28, names=['encoder','decoder'],
               name='vae',**kwargs):
    super().__init__(name=name,**kwargs)
    
    # The outer_layers
    self.encoder = VariationalEncoder(encoding_seq, latent_dim, intermediate_dim, name=names[0])
    self.decoder = VariationalDecoder(decoding_seq, intermediate_dim, name=names[1])


  def call(self, inputs):
    latent_samples, latent_mean, latent_variance = self.encoder(inputs)
    reconstructed = self.decoder(latent_samples)
    # Add KL divergence regularization loss.
    kl_loss = -0.5 * tf.reduce_mean(
        latent_variance - tf.square(latent_mean) - tf.exp(latent_variance) + 1
    )
    self.add_loss(kl_loss)
    return reconstructed


  
