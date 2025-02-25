import kagglehub
import tensorflow as tf
import matplotlib.pyplot as plt


#DOWNLOAD DATASET (kaggle)
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", path)



BATCH_SIZE = 32
IMG_SIZE = (224,224)
SEED = 123
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{path}/Train",
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE,
    shuffle = True,
    validation_split=0.2,
    subset='training',
    seed= SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{path}/Train",
    batch_size = BATCH_SIZE,
    image_size = IMG_SIZE,
    shuffle = True,
    validation_split=0.2,
    subset='validation',
    seed= SEED
)

for images, labels in train_ds.take(1):
    print("Train batch shape:", images.shape)
    print("Train labels shape:", labels.shape)

for images, labels in val_ds.take(1):
    print("Validation batch shape:", images.shape)
    print("Validation labels shape:", labels.shape)


print("Class names:", train_ds.class_names)


plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(str(train_ds.class_names[labels[i]]))
        plt.axis("off") 


normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
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
    train_ds,
    validation_data=val_ds,
    epochs=20
)


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

