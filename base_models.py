"""
Architectures for the base models that are going to be used in the project.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, backend


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

class ConvolutionSampling(layers.Layer):
    """
    A simple stack of two convolution layers, a max pooling save.
    """
    def __init__(self, filters, kernel, strides=(1,1), padding='same', 
        normalization_momentum:int=0.8, leaky_alpha=0.2, downsampling=True,
        pooling_parameters = {'pool_size':3}, **kwargs
    ):
        super().__init__(**kwargs)

        # Determine whether it is upsampling or downsampling and create the pooling layer accordingly
        if pooling_parameters:
            if downsampling:
                self.pooling = layers.MaxPooling2D(**pooling_parameters)
            else:
                self.pooling = layers.UpSampling2D(**pooling_parameters)
        else:
            self.pooling = None

        # Check whether there is any normalisation
        self.normalisation = None if normalization_momentum is None else layers.BatchNormalization(momentum=normalization_momentum)
        last_activation = None if self.normalisation else layers.LeakyReLU(alpha=leaky_alpha) 

        # Count the number of convolution layers
        # Create the convolution layers depending on whether there is a normalisation layer
        if type(filters) is int:
            strides = strides if type(strides) is int else strides[0]
            self.convolutions = [layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding, activation=last_activation)]
        else:
            self.convolutions = [ layers.Conv2D(filters=filters[i], kernel_size=kernel[i], strides=strides[i], padding=padding,
                                                activation=layers.LeakyReLU(alpha=leaky_alpha)) for i in range(0, len(filters)-1)]
            # The last convolution has no activation if we normalise the input
            self.convolutions.append(
                layers.Conv2D(filters=filters[-1], kernel_size=kernel[-1], strides=strides[-1], padding=padding, activation=last_activation)
            )

        self.activation = layers.LeakyReLU(alpha=leaky_alpha)
      
    def call(self, inputs):
        x = inputs
        for convolution_layer in self.convolutions:
            x = convolution_layer(x)
        if self.pooling:
            x = self.pooling(x)
        if self.normalisation:
            x = self.normalisation(x)
            x = self.activation(x)
        return x

    def summarize(self, input_sz):
        x = layers.Input(shape=input_sz, name='input')
        Model(x, self.call(x), name=self.name).summary()



def get_convolutional_encoder(input_dim):
    """
    Creates a three-block convolutional network based on the ConvolutionSampling class
    """
    input_img = keras.Input(shape=input_dim)
    x = ConvolutionSampling(filters=(16), kernel=(3), normalization_momentum=None,
            downsampling=True, pooling_parameters={'pool_size':2, 'padding':'same'})(input_img)
    x = ConvolutionSampling(filters=8, kernel=3, normalization_momentum=None,
                downsampling=True, pooling_parameters={'pool_size':2, 'padding':'same'})(x)                  
    encoded = ConvolutionSampling(filters=8, kernel=3, normalization_momentum=None,
                            downsampling=True, pooling_parameters={'pool_size':2, 'padding':'same'}, name='encoded')(x)
    convolutional_encoder = Model(input_img, encoded, name='convolutional_prencoder')
    # Intermediate Dim is 16,16,8 (from 128,128,3)
    return convolutional_encoder

def get_convolutional_decoder(intermediate_dim):
    """
    Creates a three-block convolutional network based on the ConvolutionSampling class.
    The intermediate dimension is the size of the input
    """
    intermediate_x = keras.Input(shape=intermediate_dim)
    x = ConvolutionSampling(filters=8, kernel=3, normalization_momentum=None,
                            downsampling=False, pooling_parameters={'size':2, 'interpolation':'bilinear'})(intermediate_x)
    x = ConvolutionSampling(filters=8, kernel=3, normalization_momentum=None,
                            downsampling=False, pooling_parameters={'size':2, 'interpolation':'bilinear'})(x)
    x = ConvolutionSampling(filters=16, kernel=3, normalization_momentum=None,
                            downsampling=False, pooling_parameters={'size':2, 'interpolation':'bilinear'})(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded')(x)
    convolutional_decoder = Model(intermediate_x, decoded, name='convolutional_decoder')
    return convolutional_decoder


class VariationalAutoencoder(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        # Compilation parameters
        self.loss_factor = 10000
        self.reconstruction_loss_fn = keras.losses.mse

    
    def set_hyperparameters(self, recon_lossfn = keras.losses.mse, loss_factor=10000):
      self.loss_factor = 10000
      self.reconstruction_loss_fn = recon_lossfn
      
      
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
      z, _, _ = self.encoder(inputs)
      outputs = self.decoder(z)
      return outputs

    def train_step(self, data):

        # inputs, outputs = data

        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(self.reconstruction_loss_fn(data[1], reconstruction), axis=(1, 2))
            #     tf.reduce_sum( 
            #     )
            # )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            # kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss*self.loss_factor + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def encode(self, inputs):
      z, _, _ = self.encoder(inputs)
      return z
    
    def decode(self, latent_inputs):
      return self.decoder(latent_inputs)