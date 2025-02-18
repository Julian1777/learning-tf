import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError



data = pd.read_csv('train.csv')

print(data.head())
print(data.shape)

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
tensor_data = tf.random.shuffle(tensor_data)
print(tensor_data[:5])

x = tensor_data[:,3:-1]
print(x[:5])

y = tensor_data[:,-1]
y = tf.expand_dims(y, axis = -1)
print(y[:5])

normalizer = Normalization()
x_normalized = tf.constant([[3,4,5,6,7],
                            [4,5,6,7,8],
                            [32,1,56,3,5]])
print(normalizer.adapt(x_normalized))
print(normalizer(x_normalized))

print(x.shape)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(x)

x_train = x[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE)

for x,y in train_dataset:
    print(x,y)
    break

x_val = x[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

x_test = x[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

normalizer = Normalization(input_shape=(x.shape[1],))
normalizer.adapt(x_train)
print(normalizer(x)[:5])


model = keras.Sequential([
    InputLayer(input_shape = (8,)),
    normalizer,
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    layers.Dense(1)
])

print(model.summary())


tf.keras.utils.plot_model(model, to_file='model.png', show_shapes = True)

model.compile(optimizer = Adam(learning_rate = 0.1),
                loss = MeanAbsoluteError(),
                metrics = [RootMeanSquaredError()])

history = model.fit(train_dataset, validation_data=val_dataset, epochs = 100, verbose = 1)


plt.plot(history.history['loss'], label='train loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='val loss')
else:
    print("Validation loss not available.")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

history.history

plt.plot(history.history['root_mean_squared_error'])
if 'val_root_mean_squared_error' in history.history:
    plt.plot(history.history['val_root_mean_squared_error'], label='RMSE')
else:
    print("Validation loss not available.")
plt.title('Model Performance')
plt.ylabel('rmse')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
plt.show()

print('Model Evaluation')
print(model.evaluate(x,y))
print('Validation')
print(model.evaluate(x_val,y_val))
print('Testing')
print(model.evaluate(x_test,y_test))

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

print(model.predict(tf.expand_dims(x_test[0], axis=0)))

print(y_test[0])

y_true = list(y_test[:,0].numpy())

y_pred = list(model.predict(x_test)[:,0])
print(y_pred)

ind = np.arange(100)
plt.figure(figsize=(40,20))

width = 0.4

plt.bar(ind, y_pred, width, label='Predicted Car Price', color='blue')
plt.bar(ind + width, y_true, width, label='Actual Car Price', color='orange')

plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('Car Price Prices')

plt.legend()
plt.show()