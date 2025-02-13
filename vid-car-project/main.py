import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.optimizer import Adam


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

normalizer = Normalization(input_shape=(x.shape[1],))
normalizer.adapt(x)
print(normalizer(x)[:5])


model = keras.Sequential([
    InputLayer(input_shape = (8,)),
    normalizer,
    layers.Dense(1)
])

print(model.summary())


tf.keras.utils.plot_model(model, to_file='model.png', show_shapes = True)
model.compile(optimizer = Adam(), loss = MeanAbsoluteError())
history = model.fit(x,y, epochs = 100, verbose = 1)