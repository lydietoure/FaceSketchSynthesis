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


class SSIMLoss(keras.losses.Loss):
    """
    SSIM Loss as implemented in Toledo and Antonelo:
    https://github.com/tldrafael/FaceReconstructionWithVAEAndFaceMasks
    """
    def __init__(self):
        super(SSIMLoss, self).__init__(name='ssim')

    def __call__(self, y_e, y_pred, sample_weight=None):
        loss_ssim = 1 - tf.image.ssim(y_e, y_pred, 1.)
        # loss_ssim = tf.math.reduce_mean(loss_ssim)
        return tf.math.reduce_mean(loss_ssim)


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


def convolutionStack(
    x: layers.Layer, filters, kernel_size=3, activation=None,
    strides=1, padding='same', use_bias=True, k_initializer='glorot_uniform',
    use_bn=False, use_drop=False, drop_value=0.3, upsampling=False,
    ):
    """
    Creates a stack of Conv2D-BatchNormalisation-Activation-Dropout with the given parameters.
    """

    if upsampling:
        x = layers.Conv2DTranspose(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, 
            kernel_initializer=k_initializer
        )(x)
    else:
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, 
            kernel_initializer=k_initializer
        )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_drop:
        x = layers.Dropout(drop_value)(x)
    return x

class Autoencoder(Model):
    def __init__(self, encoder, decoder, latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim if latent_dim else tuple(self.encoder.layers[-1].output.shape[1:])
        self.input_size = tf.shape(self.encoder.layers[0].output.shape[1:])

    def call(self, inputs):
        return self.decoder(self.encoder(inputs))



def get_autoencoder(input_shape, model_name:str = 'autoencoder'):
    """
    A simple convolutional autoencoder.
    """
    # Inputs
    input_img = layers.Input(shape=input_shape, name='input_image')

    # Encoder
    x = convolutionStack(input_img, 32, activation=layers.LeakyReLU(0.2), 
            kernel_size=3, use_bn=False, strides=2)
    x = convolutionStack(x, 32, activation=layers.LeakyReLU(0.2), 
            kernel_size=3, use_bn=False, strides=1)
    latent_inputs = convolutionStack(x, 16, activation=layers.LeakyReLU(0.2), 
            kernel_size=3, strides=2)
    encoder = Model(x, latent_inputs)
    latent_inputs_shape = latent_inputs.shape[1:]

    # Decoder
    latent_x = layers.Input(shape=latent_inputs_shape)
    x = layers.UpSampling2D(size=2, interpolation='bilinear')(latent_x)
    x = convolutionStack(x, 32, activation=layers.LeakyReLU(0.2), 
            kernel_size=3, use_bn=False, strides=1)
    x = convolutionStack(x, 16, activation=layers.LeakyReLU(0.2), 
            kernel_size=3, use_bn=True, strides=1)
    x = layers.UpSampling2D(size=2, interpolation='lanczos5')(x)
    x = layers.Conv2D(3, kernel_size=3, strides=1, padding='same', 
            activation='sigmoid', name='decoded_output')(x)
    decoder = Model(latent_x, x)
    
    autoencoder = Autoencoder(encoder, decoder, latent_inputs_shape, name=model_name)
    return autoencoder

class VariationalAutoencoder(Autoencoder):
    
    def __init__(self, encoder, decoder, latent_dim, reparametrized:bool=True, **kwargs):
        """"
        Creates a variational autoencoder for real-valued inputs/outputs.
    
        A variational autoencoder is a semi-supervised learning model which describes
        data generation through a probabilistic distribution of some latent (unobserved) samples 
        conditioned on the observed input. 
        It is composed of two subnetworks: a probabilistic encoder (called recognition model)
        which approximates a posterior distribution of the latent space conditioned on the input,
        and a probabilistic decoder (called generative model) learns the conditional distribution 
        of the input conditioned on the latent samples. 

        This variable autoencoder model assumes real-valued inputs, and uses the standard Gaussian
        distribution as the prior distribution of the latent samples.
        The encoder to supply is required to have a Dense layer as final layer,
        and the input layer of the decoder must 

        Params:
        - encoder: the recognition model which takes some input `x` and encodes it 
        - decoder: the generative model, which generates an output `y` from a latent representation `z` coded by the recognition model
        - latent_dim: the size of the latent dimension from which latent samples will be drawn
        - reparametrized: whether the encoder also reparametrizes the latent samples. In that case, the encode outputs z,z_mean,z_log_var
        """
        super().__init__(encoder, decoder, latent_dim, **kwargs)
        # self.encoder = encoder
        # self.decoder = decoder
        self.input_size = tf.shape(self.encoder.layers[0].output.shape[1:])
        # self.latent_dim = latent_dim
        self.reparametrized = reparametrized

        # Verify consistency
        # decoder_output_size = tf.shape(self.decoder.layers[-1].output.shape[1:])
        # decoder_input_size = tf.shape(self.decoder.layers[0].output.shape[1:])
        # assert decoder_output_size == self.input_size, 'Inputs and output sizes not compatible'
        # assert decoder_input_size == self.latent_dim, 'The dimension of the latent space is not compatible with the input of the decoder'

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.distribution = layers.Dense(latent_dim)
      
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def compile(self, rec_loss_factor=10, kl_loss_factor=1, **kwargs):
        super(VariationalAutoencoder, self).compile(**kwargs)
        self.reconstruction_loss_factor = rec_loss_factor
        self.kl_loss_factor = kl_loss_factor

    def reparametrize(self,inputs):
      distribution_mean, distribution_variance = inputs
      batch_size = backend.shape(distribution_variance)[0]
      random = backend.random_normal(shape=(batch_size, backend.shape(distribution_variance)[1]))
      latent_samples = distribution_mean + backend.exp(0.5 * distribution_variance) * random
      return latent_samples

    def call(self, inputs):
      z,_,_ = self.encode(inputs)
      outputs = self.decoder(z)
      return outputs

    def train_step(self, data):
        inputs, outputs = data

        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encode(inputs)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(self.loss(outputs, reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss * self.reconstruction_loss_factor + kl_loss * self.kl_loss_factor
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
        if self.reparametrized:
            return self.encoder(inputs)
        
        flattened_encoding = self.encoder(inputs)
        z_mu = self.distribution(flattened_encoding)
        z_log_var = self.distribution(flattened_encoding)
        z = self.reparametrize((z_mu, z_log_var))
        return z, z_mu, z_log_var

    def get_random_latent_vectors(self, batch_size):
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        return z
    
    def decode(self, latent_inputs):
        return self.decoder(latent_inputs)


