import tensorflow as tf
import numpy as np
import tensorflow
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, Normalization, BatchNormalization
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError





#LOADING DATASET AND SPLITTING INTO TRAIN VAL AND TEST

dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, shuffle_files=True, split=['train'])



def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
    DATASET_SIZE = len(dataset)

    train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))
    val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
    val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))
    test_dataset = val_test_dataset.skip(int(VAL_RATIO*DATASET_SIZE))

    return train_dataset, val_dataset, test_dataset



TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

#dataset = tf.data.Dataset.range(10)
train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print(list(train_dataset.take(1).as_numpy_iterator()), list(val_dataset.take(1).as_numpy_iterator()), list(test_dataset.take(1).as_numpy_iterator()))



#DATASET VISUALIZATION

for i, (image,label) in enumerate(train_dataset.take(16)):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(image)
    plt.title(dataset_info.features['label'].int2str(label))
    plt.axis('off')

plt.show()


#DATA PROCESSING
IM_SIZE = 224
def resize_rescale(image, label):
    return tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

for image,label in train_dataset.take(1):
    print(image, label)


BATCH_SIZE = 32

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#MODEL CREATION

normalizer = Normalization()

lenet_model = tf.keras.Sequential([
    InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),

    Conv2D(filters = 6, kernel_size=3, strides=1, padding='valid'),
    BatchNormalization(),
    layers.Activation('relu'),
    MaxPool2D(pool_size = 2, strides = 2),

    Conv2D(filters = 16, kernel_size=3, strides=1, padding='valid'),
    BatchNormalization(),
    layers.Activation('relu'),
    MaxPool2D(pool_size = 2, strides = 2),

    Flatten(),

    Dense(100, activation="relu"),
    BatchNormalization(),
    Dense(10, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="sigmoid"),
])

print(lenet_model.summary())
tf.keras.utils.plot_model(lenet_model, to_file='model.png', show_shapes = True)


lenet_model.compile(
    optimizer = Adam(learning_rate = 0.01),
    loss = BinaryCrossentropy(),
    metrics= ['accuracy']
    #metric = RootMeanSquaredError()
)

history = lenet_model.fit(train_dataset, validation_data=val_dataset, epochs = 20, verbose = 1)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss'])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train_accuracy', 'val_accuracy'])


plt.show()


#MODEL EVAL
print(lenet_model.evaluate(test_dataset))

def parasitized_or_not(x):
    if x < 0.5:
        return str('Parisitized')
    else:
        return str('Uninfected')

for images, labels in test_dataset.take(1):
    preds = lenet_model.predict(images)
    print("Prediction for first image:", parasitized_or_not(preds[0][0]))


for i, (image, label) in enumerate(test_dataset.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image[0])
    plt.title(str(parasitized_or_not(label.numpy()[0])) + ":" + str(parasitized_or_not(lenet_model.predict(image)[0][0])))

    plt.axis('off')