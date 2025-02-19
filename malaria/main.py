import tensorflow as tf
import numpy as np
import tensorflow
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, Rescaling, Resizing, Normalization
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    return tf.image.resize(image, (IM_SIZE, IM_SIZE))/ 225.0, label

train_dataset = train_dataset.map(resize_rescale)

for image,label in train_dataset.take(1):
    print(image, label)

train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

#MODEL CREATION

normalizer = Normalization()

model = tf.keras.Sequential([
    InputLayer(input_shape = (IM_SIZE, IM_SIZE, 3)),

    Conv2D(filters = 6, kernel_size=5, strides=1, padding='valid', activation='sigmoid'),
    MaxPool2D(pool_size = 2, strides = 2),

    Conv2D(filters = 16, kernel_size=5, strides=1, padding='valid', activation='sigmoid'),
    MaxPool2D(pool_size = 2, strides = 2),

    Flatten(),

    Dense(100, activation="sigmoid"),

    Dense(10, activation="sigmoid"),

    Dense(2, activation="sigmoid"),
])

print(model.summary())