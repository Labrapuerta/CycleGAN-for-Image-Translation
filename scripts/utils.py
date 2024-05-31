from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [256, 256]

def apply_pca_and_visualize(convolved_images):
    # Flatten the convolved images
    shape = convolved_images.shape
    normalized_image = convolved_images / 255.0
    reshaped_image = tf.reshape(normalized_image, (-1, shape[-1]))
    reshaped_array = reshaped_image.numpy()
    # Apply PCA
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(reshaped_array)
    # Reshape back
    pca_image_reshaped = pca_result.reshape(shape[0], shape[1], 1)
    return pca_image_reshaped

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image

def load_dataset(filenames, labeled=True, ordered=False, repeat = False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    if repeat:
        dataset = dataset.repeat(count = 20)
    dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset