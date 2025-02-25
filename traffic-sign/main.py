import kagglehub
import tensorflow as tf
import tensorflow
import matplotlib.pyplot as plt
import hashlib
import json
import os
from tensorflow.keras.models import load_model

def get_model_hash(model):
    model_json = model.to_json()
    return hashlib.md5(model_json.encode()).hexdigest()

def save_model_with_hash(model, model_path="traffic_sign_model.h5", hash_path="model_hash.txt"):
    model.save(model_path)
    model_hash = get_model_hash(model)
    with open(hash_path, "w") as f:
        f.write(model_hash)

def load_model_if_valid(model_path="traffic_sign_model.h5", hash_path="model_hash.txt"):
    if not os.path.exists(model_path) or not os.path.exists(hash_path):
        return None
    model = load_model(model_path)
    with open(hash_path, "r") as f:
        saved_hash = f.read().strip()
    if get_model_hash(model) == saved_hash:
        print("Loaded saved model.")
        return model
    else:
        print("Model architecture changed! Retraining...")
        return None

path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Path to dataset files:", path)

BATCH_SIZE = 32
IMG_SIZE = (224,224)
SEED = 123
train_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{path}/Train",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=SEED
)

val_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{path}/Train",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

val_batches = int(0.5 * len(val_test_ds))
val_ds = val_test_ds.take(val_batches)
test_ds = val_test_ds.skip(val_batches)

print(f"Train dataset size: {len(train_val_ds)} batches")
print(f"Validation dataset size: {len(val_ds)} batches")
print(f"Test dataset size: {len(test_ds)} batches")

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_val_ds = train_val_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = load_model_if_valid()
if not model:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(43, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history = model.fit(
        train_val_ds,
        validation_data=val_ds,
        epochs=20
    )
    save_model_with_hash(model)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plt.figure(figsize=(10, 10))
for images, labels in train_val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(str(train_val_ds.class_names[labels[i]]))
        plt.axis("off")

for images, labels in val_ds.take(1):  
    predictions = model.predict(images)
    predicted_classes = tf.argmax(predictions, axis=1)
    for i in range(5):
        plt.figure(figsize=(4, 4))
        plt.imshow(images[i].numpy().astype("uint8"))   
        plt.title(f"True: {val_ds.class_names[labels[i]]}\nPredicted: {val_ds.class_names[predicted_classes[i]]}")
        plt.axis("off")
plt.show()