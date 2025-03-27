import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('car_train2.csv', )
print(type(data))
print("Train Columns:", data.columns)

print(data.head())
print(data.shape)

sns.pairplot(data[['years', 'km', 'rating', 'condition', 'economy', 'top speed', 'hp', 'torque', 'current price']], diag_kind='kde')
#plt.show()

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
print(tensor_data[:5])
tensor_data = tf.random.shuffle(tensor_data)
print(tensor_data[:5])


x = tensor_data[:,3:-1]
print(x[:5])

y = tensor_data[:-1]
y = tf.expand_dims(y, axis = -1)
print(y[:5])