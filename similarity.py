from image_similarity_measures.quality_metrics import ssim, fsim, psnr
from piq import FID
from torch import from_numpy
from keras.losses import mse

import tensorflow as tf
import numpy as np



def fid(y_true, y_pred):
    """
    Computes the Frechet Inception distance between two images given as numpy arrays
    """
    assert y_true.shape == y_pred.shape, 'The inputs must have the same size'
    
    sh = y_true.shape
    w,h = sh[0], sh[1]

    if len(sh) == 3:
        # Convert to grayscale
        img1 = np.array(tf.image.rgb_to_grayscale(y_true)).reshape(w,h)
        img2 = np.array(tf.image.rgb_to_grayscale(y_pred)).reshape(w,h)

    img1 = from_numpy(img1)
    img2 = from_numpy(img2)
    fid_metric = FID()
    
    index = fid_metric(img1, img2)
    return float(index)
    


PRE_LOADED_MEASURES = {
    'psnr': psnr,
    'ssim': ssim,
    'fsim': fsim,
    'fid': fid,
    'mse': mse
}

class Evaluator:
    """
    A class which evaluates a model with respect to a number of similarity measures
    """

    def __init__(self, name):
        self.name = name
        self.results = {}

    def evaluate_model_on_batch(self, y_true, measure, model:tf.keras.Model):
        # self.model = model if self.model is None else self.model
        # assert model, "No model was assigned to be evaluated"

        y_pred = model.predict(y_true)
        return self.evaluate_batch(y_true, y_pred, measure)

    def evaluate_batch(self, y_true, y_pred, measure, record=True):
        """Evaluates a batch of data wrt a given similarity measure, and returns the mean value of the similarity indices"""
        
        # assert len(y_t)

        if isinstance(measure, str):
            measure = PRE_LOADED_MEASURES[measure.lower()]
        
        similarity_values = [ measure(y_true[i], y_pred[i]) for i in range(len(y_true)) ]
        similarity_aggregate = float(tf.reduce_mean(similarity_values))

        if record:
            measure_name = measure.__name__
            self.results[measure_name] = similarity_aggregate

        return similarity_aggregate
     