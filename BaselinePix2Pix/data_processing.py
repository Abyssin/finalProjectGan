import tensorflow as tf
import os
import numpy as np
from PIL import Image


# Constants - following original pix2pix setup
BUFFER_SIZE = 400
BATCH_SIZE = 1  # Following original pix2pix paper for better U-Net results
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_image_pair_from_processed_data(image_path, mask_path):
    """
    Load preprocessed image and mask pair from your processed_data structure using PIL.
    
    Uses PIL to load TIFF files (which TensorFlow doesn't natively support).
    """
    def load_tiff_image(path):
        # Convert path tensor to string if needed
        if hasattr(path, 'numpy'):
            path = path.numpy().decode('utf-8')
        elif isinstance(path, bytes):
            path = path.decode('utf-8')
        
        # Load with PIL and convert to numpy array
        pil_image = Image.open(path).convert('L')  # Ensure grayscale
        np_image = np.array(pil_image, dtype=np.uint8)
        
        # Add channel dimension and ensure shape [286, 286, 1]
        np_image = np.expand_dims(np_image, axis=-1)
        
        return np_image.astype(np.float32)
    
    # Use tf.py_function to call PIL loading
    image = tf.py_function(load_tiff_image, [image_path], tf.float32)
    mask = tf.py_function(load_tiff_image, [mask_path], tf.float32)
    
    # Set shapes explicitly
    image.set_shape([286, 286, 1])
    mask.set_shape([286, 286, 1])
    
    return image, mask


def random_crop_from_286(input_image, real_image):
    """
    Random crop from 286x286 to 256x256.
    
    Only crops down (as your images are already 286x286), doesn't upscale.
    """
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])

    return cropped_image[0], cropped_image[1]


def resize_to_256(input_image, real_image):
    """
    Resize to 256x256 - following original pix2pix approach for test/val.
    
    This is what the original pix2pix does for test/validation data.
    Unlike your current GAN which center crops, this resizes.
    """
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [IMG_HEIGHT, IMG_WIDTH],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return input_image, real_image


def normalize_for_baseline_gan(input_image, real_image):
    """
    Normalize images for baseline GAN training.
    
    Modified for grayscale input and binary mask output:
    - Input images: normalized to [-1, 1] (grayscale)
    - Target masks: normalized to [0, 1] and binarized (for sigmoid output)
    """
    # Normalize input image to [-1, 1] 
    input_image = (input_image / 127.5) - 1
    
    # Normalize mask to [0, 1] and ensure binary
    real_image = real_image / 255.0
    real_image = tf.cast(real_image > 0.5, tf.float32)
    
    return input_image, real_image


@tf.function()
def random_jitter_from_processed(input_image, real_image):
    """
    Apply random jittering to preprocessed 286x286 data for training.
    
    Follows your approach: crop down from 286x286 to 256x256, then random flip.
    """
    # Random crop from 286x286 to 256x256 (your images are already 286x286)
    input_image, real_image = random_crop_from_286(input_image, real_image)

    # Random horizontal flip (matches original pix2pix)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train_processed(image_path, mask_path):
    """
    Load and preprocess training images from your processed_data structure.
    """
    input_image, real_image = load_image_pair_from_processed_data(image_path, mask_path)
    input_image, real_image = random_jitter_from_processed(input_image, real_image)
    input_image, real_image = normalize_for_baseline_gan(input_image, real_image)

    return input_image, real_image


def load_image_test_processed(image_path, mask_path):
    """
    Load and preprocess test/validation images from your processed_data structure.
    
    IMPORTANT: Follows original pix2pix approach - RESIZES test/val data to 256x256
    instead of cropping. This is different from your current GAN which crops all data.
    """
    input_image, real_image = load_image_pair_from_processed_data(image_path, mask_path)
    input_image, real_image = resize_to_256(input_image, real_image)  # RESIZE, don't crop
    input_image, real_image = normalize_for_baseline_gan(input_image, real_image)

    return input_image, real_image


def create_dataset_from_processed_data(data_dir, split='train'):
    """
    Create TensorFlow dataset from your processed_data structure.
    """
    images_dir = os.path.join(data_dir, split, 'images')
    masks_dir = os.path.join(data_dir, split, 'masks')
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.tif') and 'image' in f]
    image_files.sort()
    
    # Create corresponding mask file paths
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f.replace('_image.tif', '_mask.tif')) for f in image_files]
    
    # Verify all mask files exist
    for mask_path in mask_paths:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    print(f"Found {len(image_files)} {split} samples in {data_dir}")
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Apply appropriate preprocessing
    if split == 'train':
        dataset = dataset.map(load_image_train_processed, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(BUFFER_SIZE)
    else:
        # For test/val: RESIZE instead of crop (following original pix2pix)
        dataset = dataset.map(load_image_test_processed,
                            num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(BATCH_SIZE)
    
    return dataset