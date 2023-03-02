# Parameter Tuning

http://louistiao.me/listings/vae/variational_autoencoder.ipynb.html

## Model 1 : Convolutional Autoencoder

## Model 2: Variational Autoencoder
Architecture
    Input
    Conv2D
    BatchNorm1?
    LeakyReLU
    Conv2D
    BatchNorm2?

Hyperparameters:
    batchNorm1: True
    batchNorm2: False
    batchNorm3: True
    latent_dim: 512
    batchNorm4: True
    up_interpolation1: lanczos5
    batchNorm5: False
    up_interpolation2: bicubic
    lr: 0.0006805386138116091
    shuffle: True
    batch_size: 32


## Model 3:
Best reconstruction_loss So Far: 0.9823714137077332
Hyperparameters:
    latent_dim: 368
    bn_enc1: True
    bn_enc2: False
    bn_enc3: True
    bn_decc1: True
    up_interpolation1: bicubic
    bn_dec2: False
    up_interpolation2: lanczos3
    drop_dec3: True
    bn_conv1: False
    bn_conv2: False
    drop_conv3: False
    bn_conv3: False
    conv_units: 192
    lr_ae: 0.0006574428982248856
    lr_cv: 0.0016370333999706642
    shuffle: False
    batch_size: 80
    Score: 0.9823714137077332

		Total elapsed time: 01h 56m 17s
		Results summary
		Results in /content/drive/Othercomputers/LION-YG7/_SCC413/FaceSketchSynthesis/models/tuner_repae
		Showing 3 best trials
		<keras_tuner.engine.objective.Objective object at 0x7f6d340ea490>
		Trial summary
	
		Trial summary
		Hyperparameters:
		latent_dim: 368
		bn_enc1: False
		bn_enc2: False
		bn_enc3: False
		bn_decc1: False
		up_interpolation1: lanczos3
		bn_dec2: True
		up_interpolation2: lanczos3
		drop_dec3: False
		bn_conv1: False
		bn_conv2: True
		drop_conv3: False
		bn_conv3: False
		conv_units: 64
		lr_ae: 0.0069405535544335905
		lr_cv: 0.00016362252883080652
		shuffle: True
		batch_size: 64
		Score: 0.9954017996788025
		time: 1h 44min 25s