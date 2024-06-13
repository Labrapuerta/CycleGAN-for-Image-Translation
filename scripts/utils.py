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

### TF dataset
def preprocess_image(image, target_size = (256,256)):
    image =  smart_resize(image, target_size)
    return image

def load_and_preprocess_image(file_path, channels, target_size=(256, 256) ):
    image = tf.io.read_file(file_path)
    if channels == 1:
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.image.grayscale_to_rgb(image)
    else:
        image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)   
    image = preprocess_image(image, target_size)
    return image

def create_dataset(directory, target_size=(256, 256), channels = 3):
    dataset = tf.io.gfile.glob(str(f'{directory}/*'))
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(lambda x: load_and_preprocess_image(x, channels, target_size), num_parallel_calls=AUTOTUNE)
    return dataset

### Visualize the training
class callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        ### Every 5 epochs plot the generator
        randomint = np.random.randint(0,8,1)[0]
        if epoch != 0:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
            fig.suptitle(f'Generator evolution epoch #{epoch}', fontsize=16)
            ax = axes[0]
            ax.imshow(test_ct[randomint])
            axes[0].set_title('Original CT', size='large', loc='center')
            ax.axis('off')
            ax = axes[1]
            ax.imshow(cycle_gan.mri_generator_(test_ct)[randomint])
            axes[1].set_title('CT to MRI', size='large', loc='center')
            ax.axis('off')
            ax = axes[2]
            ax.imshow(test_mri[randomint])
            axes[2].set_title('Original MRI', size='large', loc='center')
            ax.axis('off')
            ax = axes[3]
            ax.imshow(cycle_gan.ct_generator_(test_mri)[randomint])
            axes[3].set_title('MRI to CT', size='large', loc='center')
            ax.axis('off')
            plt.show()
        if epoch == 0:
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
            fig.suptitle(f'Starting point', fontsize=16)
            ax = axes[0]
            ax.imshow(test_ct[randomint])
            axes[0].set_title('Original CT', size='large', loc='center')
            ax.axis('off')
            ax = axes[1]
            ax.imshow(cycle_gan.mri_generator_(test_ct)[randomint])
            axes[1].set_title('CT to MRI', size='large', loc='center')
            ax.axis('off')
            ax = axes[2]
            ax.imshow(test_mri[randomint])
            axes[2].set_title('Original MRI', size='large', loc='center')
            ax.axis('off')
            ax = axes[3]
            ax.imshow(cycle_gan.ct_generator_(test_mri)[randomint])
            axes[3].set_title('MRI to CT', size='large', loc='center')
            ax.axis('off')
            plt.show()
            
