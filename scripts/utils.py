from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.image import resize_with_pad
import tensorflow as tf
import tensorflow as tf
import shutil
from PIL import Image
import numpy as np
import kaggle
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

def apply_pca_and_visualize(convolved_images):
    shape = convolved_images.shape
    normalized_image = convolved_images / 255.0
    reshaped_image = tf.reshape(normalized_image, (-1, shape[-1]))
    reshaped_array = reshaped_image.numpy()
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(reshaped_array)
    pca_image_reshaped = pca_result.reshape(shape[0], shape[1], 1)
    return pca_image_reshaped

def download_dataset(dataset_name = 'darren2020/ct-to-mri-cgan'):
    kaggle.api.authenticate()
    try:
        shutil.rmtree('data')
    except:
        pass
    os.mkdir('data')
    kaggle.api.dataset_download_files(dataset_name, path='data', unzip=True)
    os.mkdir("data/Dataset/images/train")
    os.mkdir("data/Dataset/images/test")
    shutil.move("data/Dataset/images/trainA", "data/Dataset/images/train/")
    shutil.move("data/Dataset/images/trainB", "data/Dataset/images/train/")
    shutil.move("data/Dataset/images/testA", "data/Dataset/images/test/")
    shutil.move("data/Dataset/images/testB", "data/Dataset/images/test/")
    os.rename("data/Dataset/images/train/trainA", "data/Dataset/images/train/CT")
    os.rename("data/Dataset/images/train/trainB", "data/Dataset/images/train/MRI")
    os.rename("data/Dataset/images/test/testA", "data/Dataset/images/test/CT")
    os.rename("data/Dataset/images/test/testB", "data/Dataset/images/test/MRI")
    os.rename("data/Dataset/unseen_demo_images/ct", "data/Dataset/unseen_demo_images/CT")
    os.rename("data/Dataset/unseen_demo_images/mri", "data/Dataset/unseen_demo_images/MRI")

def photo_mean_size(image_dir):
    heights = []
    widths = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(image_dir, filename)
            with Image.open(filepath) as img:
                channels = len(img.getbands())
                width, height = img.size
                widths.append(width)
                heights.append(height)
    
    mean_height = np.mean(heights)
    mean_width = np.mean(widths)
    return int(mean_width), int(mean_height), channels

def preprocess_image(image, target_size = (256,256)):
    image =  smart_resize(image, target_size)
    image /= 255.0
    return image

def load_and_preprocess_image(file_path, target_size=(256, 256)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = preprocess_image(image, target_size)
    return image

def create_dataset(directory, target_size=(256, 256), batch_size=36):
    dataset = tf.data.Dataset.list_files(f'{directory}/*') 
    dataset = dataset.map(lambda x: load_and_preprocess_image(x, target_size), num_parallel_calls=AUTOTUNE)
    return dataset
