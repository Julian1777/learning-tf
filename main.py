import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

dftrain = pd.read_csv('car_train.csv')
dfeval = pd.read_csv('car_eval.csv')

y_train = dftrain.pop('Price')
y_eval = dfeval.pop('Price')

categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
numeric_columns = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']

feature_columns = []

for feature_name in categorical_columns:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(len(categorical_columns) + len(numeric_columns),)),
    feature_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # Use 'sigmoid' for binary classification
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

def make_dataset(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data_df))
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds

train_ds = make_dataset(dftrain, y_train, num_epochs=10, shuffle=True, batch_size=32)
eval_ds = make_dataset(dfeval, y_eval, num_epochs=1, shuffle=False, batch_size=32)

steps_per_epoch = len(dftrain) // 32
model.fit(train_ds, epochs=10, steps_per_epoch=steps_per_epoch)

eval_steps = len(dfeval) // 32
result = model.evaluate(eval_ds, steps=eval_steps)
print("Accuracy:", result[1])
