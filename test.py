import networks

from keras import layers


####### DEEP ##################
intermediate_dim = 56
img_dim = 96*96*3

deep_encoder = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(None,img_dim)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu')
], name='deep_encoder')
deep_encoder.summary()

deep_decoder = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(None,intermediate_dim)),
    layers.Dense(128, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(img_dim, activation='sigmoid')
], 'deep_decoder')
deep_decoder.summary()

deep_vae
