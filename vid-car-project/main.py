import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Normalization
from tensorflow.keras.layers import Normalization


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

normalizer = Normalization(mean = 5, variance = 4)
x_normalized = tf.constant([[3,4,5,6,7]])
normalizer(x_normalized)
print(x_normalized)