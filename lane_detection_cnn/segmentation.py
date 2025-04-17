import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import os
import shutil
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CULANE_DIR = os.path.join(SCRIPT_DIR, "dataset", "culane")
IMAGES_DIR = os.path.join(CULANE_DIR, "images", "lane")
ANNOTATIONS_DIR = os.path.join(CULANE_DIR, "annotations")
MASKS_DIR = os.path.join(CULANE_DIR, "masks")
MODEL_PATH = "lane_detection_model.h5"

os.makedirs(CULANE_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


IMG_SIZE = (224, 224)
BATCH_SIZE = 128
SEED = 123
EPOCHS = 10



#Later for test images
def img_preprocessing(image_path):
    img = cv.imread(image_path)
    img = cv.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def load_image_mask_pair(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask > 127, tf.float32)  # Binary mask: 0 or 1

    return image, mask

def get_dataset(images_dir, masks_dir):
    image_paths = sorted([
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.endswith(('.jpg', '.png'))
    ])
    
    mask_paths = sorted([
        os.path.join(masks_dir, f) 
        for f in os.listdir(masks_dir) 
        if f.endswith('.png')
    ])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image_mask_pair, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def process_dataset():

    def draw_lane_mask(anno_path, image_shape):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)  # H x W
        with open(anno_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                coords = list(map(float, line.strip().split()))
                points = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
                for i in range(1, len(points)):
                    cv.line(mask, points[i-1], points[i], color=255, thickness=2)
        return mask

    base_dir = os.path.join(SCRIPT_DIR, "dataset", "driver_161_90frame")

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found at {base_dir}")
    

    output_dir = CULANE_DIR
    images_dir = IMAGES_DIR
    annotations_dir = ANNOTATIONS_DIR

    lane_class_dir = os.path.join(images_dir)

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

            img = cv.imread(img_path)
            mask = draw_lane_mask(anno_path, img.shape)
            mask = cv.resize(mask, IMG_SIZE)
            mask_output_path = os.path.join(MASKS_DIR, f"img_{img_count}.png")
            cv.imwrite(mask_output_path, mask)
            
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
        "masks_dir": MASKS_DIR,
        "count": img_count
    }


def create_unet_model(input_shape= (224, 224, 3)):
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

    # First decoder block
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    # UpSampling chosen instead of TransposedConv2D to avoid checkerboard artifacts
    
    up5 = layers.Conv2D(256, 2, activation='relu', padding='same')(up5)
    # Reducing filters (256) as we go up to match corresponding encoder block
    
    merge5 = layers.concatenate([conv3, up5], axis=3)
    # Skip connection - concatenates features from encoder to recover spatial information
    # This is the key innovation of U-Net that helps preserve fine details
    
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    # Second decoder block
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.Conv2D(128, 2, activation='relu', padding='same')(up6)
    merge6 = layers.concatenate([conv2, up6], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    # Third decoder block
    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.Conv2D(64, 2, activation='relu', padding='same')(up7)
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    # 1x1 convolution to map to a single output channel
    # Sigmoid activation for binary segmentation (0-1 range for each pixel)
    # This produces a lane probability mask

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    
    return model


lane_images_path = os.path.join(IMAGES_DIR)

image_files = [f for f in os.listdir(lane_images_path) if f.endswith(('.jpg', '.png'))]
mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]
annotaion_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.txt')]

print(f"Lane images folder: {lane_images_path} contains {len(image_files)} images.")
print(f"Annotations folder: {ANNOTATIONS_DIR} contains {len(annotaion_files)} annotaions.")
print(f"Masks folder: {MASKS_DIR} contains {len(mask_files)} annotaions.")

if os.path.exists(lane_images_path) and os.path.exists(ANNOTATIONS_DIR) and len(image_files) == len(mask_files) and len(mask_files) > 0:
    print(f"Dataset already processed. Found {len(image_files)} images and {len(mask_files)} masks.")
    processed_data = {
        "base_dir": CULANE_DIR,
        "images_dir": IMAGES_DIR,
        "annotations_dir": ANNOTATIONS_DIR,
        "count": len(image_files)
    }
else:
    print("Running dataset processing...")
    processed_data = process_dataset()
    

DATASET_PATH = processed_data["images_dir"]
print(f"Updated DATASET_PATH to: {DATASET_PATH}")

full_dataset = get_dataset(IMAGES_DIR, MASKS_DIR)

# 80-10-10 split
dataset_size = processed_data["count"]
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

full_dataset = full_dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=False)

train_ds = full_dataset.take(train_size)
val_ds = full_dataset.skip(train_size).take(val_size)
test_ds = full_dataset.skip(train_size + val_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


print(f"Train dataset size: {len(train_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")


model = create_unet_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
)

if os.path.exists(MODEL_PATH):
    print(f"Loading existing model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    print("No existing model found. Training a new model.")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS,
    )

    model.save("lane_detection_model.h5")
