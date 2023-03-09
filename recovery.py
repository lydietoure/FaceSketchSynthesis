"""
Recover learning models from zip files
"""

import os
from zipfile import ZipFile
from keras.models import load_model
from keras.preprocessing.image import save_img

# Unzip
def recover_from_zip(src_dir, target_dir, model_dir):
    """
    Extracts the Zip file at path src_dir/model_dir on the `target_dir` directory
    and then loads the model `model_dir` 
    """
    zip_path = os.path.join(src_dir, model_dir)
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    model = load_model(model_dir)
    return model

def get_available_zipped_models(models_path='./models'):
    """
    Returns the zipped models available in the models directory
    """
    models = os.listdir(models_path)  # all files
    models = [ m for m in models if os.path.splitext(m)[1]=='.zip']

    return models


def save_generated_images(images, target_dir, filenames):
    for img,fn in zip(images, filenames):
        fp = os.join.path(target_dir, fn)
        save_img(fp, img)
