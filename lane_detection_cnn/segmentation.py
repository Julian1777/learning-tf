import tensorflow as tf
import tensorflow
from tensorflow.keras import layers
import cv2 as cv
import numpy as np
import os
import shutil
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset", "culane")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
DATASET_PATH = IMAGES_DIR

IMG_SIZE = (224, 224)
BATCH_SIZE = 128
SEED = 123

print("Path to dataset:", DATASET_PATH)

#Later for test images
def img_preprocessing(image_path):
    img = cv.imread(image_path)
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def process_dataset():
    base_dir = os.path.join(SCRIPT_DIR, "dataset", "driver_161_90frame")

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found at {base_dir}")
    

    output_dir = OUTPUT_DIR
    images_dir = IMAGES_DIR
    annotations_dir = ANNOTATIONS_DIR

    lane_class_dir = os.path.join(images_dir, "lane")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(lane_class_dir, exist_ok=True)

    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders in dataset")

    img_count = 0
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        files = os.listdir(subfolder)
        
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        img_files = [f for f in files if any(f.endswith(ext) for ext in img_extensions)]
        
        print(f"Found {len(img_files)} images in {os.path.basename(subfolder)}")
        
        
        for img_file in img_files:
            img_path = os.path.join(subfolder, img_file)
            img_basename, img_ext = os.path.splitext(img_file)
            
            anno_file = f"{img_basename}.lines.txt"
            anno_path = os.path.join(subfolder, anno_file)
            
            if not os.path.exists(anno_path):
                print(f"Warning: No annotation found for {img_path}")
                continue
            
            img_count += 1
            new_img_name = f"img_{img_count}{img_ext}"
            new_anno_name = f"img_{img_count}_anno.txt"
            
            shutil.copy(img_path, os.path.join(lane_class_dir, new_img_name))
            shutil.copy(anno_path, os.path.join(annotations_dir, new_anno_name))
            
            if img_count % 100 == 0:
                print(f"Processed {img_count} images so far")
    
    print(f"Processing complete. Organized {img_count} image-annotation pairs.")
    
    copied_images = len(os.listdir(images_dir))
    copied_annos = len(os.listdir(annotations_dir))
    print(f"Files in output directories: {copied_images} images, {copied_annos} annotations")
    
    return {
        "base_dir": output_dir,
        "images_dir": images_dir,
        "annotations_dir": annotations_dir,
        "count": img_count
    }

min_files_threshold = 10

if (os.path.exists(IMAGES_DIR) and 
    os.path.exists(ANNOTATIONS_DIR) and
    len(os.listdir(IMAGES_DIR)) > min_files_threshold and
    len(os.listdir(ANNOTATIONS_DIR)) > min_files_threshold):
    
    print(f"Dataset already processed. Found {len(os.listdir(IMAGES_DIR))} images and {len(os.listdir(ANNOTATIONS_DIR))} annotations.")
    processed_data = {
        "base_dir": OUTPUT_DIR,
        "images_dir": IMAGES_DIR,
        "annotations_dir": ANNOTATIONS_DIR,
        "count": len(os.listdir(IMAGES_DIR))
    }
else:
    print("Running dataset processing...")
    processed_data = process_dataset()


def create_unet_model(input_shape(224, 224, 3)):
    #Defines the input layer. 224 pixels in height 224 pixels in width and 3 channels for RGB
    inputs = tf.keras.Input(shape=input_shape)

    #This layer learns 64 different features (edges, lines, etc) in a 3x3 grid which is the kernel
    #The relu acvitation is applied to make the model non-linear
    #The padding makes sure the size of the output stays the same
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)

    #We do the conv layer twice here to extract more features
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)

    #Max pooling will shrink the image by a factor of 2 taking the minimum value of each 2x2 grid
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    #More conv layers this time 128 filters meaning even more features are extracted
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)

    #Same again we extract even more after the conv2 layer above
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    #The same process is repeated over and over to get the most amount of features from an input

    #STILL NEEDS DECODING


    return model

DATASET_PATH = processed_data["images_dir"]
print(f"Updated DATASET_PATH to: {DATASET_PATH}")

train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE,
    shuffle = True,
    validation_split = 0.2,
    subset = "training",
    seed = SEED
)

val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed = SEED
)


train_val_ds = train_val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


val_batches = int(0.5 * len(val_test_ds))
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

print(f"Train dataset size: {len(train_val_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")
