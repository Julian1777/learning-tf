import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column as fc
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


dftrain = pd.read_csv('car_train.csv')
dfeval = pd.read_csv('car_eval.csv')

print("Train Columns:", dftrain.columns)
print("Eval Columns:", dfeval.columns)

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

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

print(result['accuracy'])  # the result variable is simply a dict of stats about our model