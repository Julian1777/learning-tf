import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv('student_performance.csv')

print('Data Columns:', data.columns)
print(data.info())
print(data.describe())


train_df, test_df = train_test_split(data, test_size = 0.2, random_state = 42)

y_train = np.log(train_df.pop('math_score'))
y_test = np.log(test_df.pop('math_score'))


categorical_columns = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

numeric_columns = ['reading_score', 'writing_score']

numeric_values_train = train_df[numeric_columns].to_numpy()

# Check for infinity values in the numeric columns
inf_count = np.isinf(numeric_values_train).sum()
print("Number of infinity values in numeric columns (train):", inf_count)

# Check for NaN values in the numeric columns
nan_count = np.isnan(numeric_values_train).sum()
print("Number of NaN values in numeric columns (train):", nan_count)

scalar = StandardScaler()

train_df[numeric_columns] = scalar.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scalar.transform(test_df[numeric_columns])


feature_columns = []

for feature_name in categorical_columns:
    vocabulary = train_df[feature_name].unique()
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
    ind_col = tf.feature_column.indicator_column(cat_col)
    feature_columns.append(ind_col)

for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

inputs = {}

for col in categorical_columns:
    inputs[col] = keras.Input(shape=(1,), name=col, dtype=tf.string)
# Numeric inputs are float32.
for col in numeric_columns:
    inputs[col] = keras.Input(shape=(1,), name=col, dtype=tf.float32)

# Use a DenseFeatures layer to process feature columns.
feature_layer = keras.layers.DenseFeatures(feature_columns)
x = feature_layer(inputs)

# Build a deeper network to improve learning capacity.
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.1)(x)
x = keras.layers.Dense(32, activation='relu')(x)
output = keras.layers.Dense(1)(x)  # Regression output: log(Price)

model = keras.Model(inputs = inputs, outputs = output)


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])


train_inputs = {name: np.array(value) for name, value in train_df.items()}
test_inputs = {name: np.array(value) for name, value in test_df.items()}

print('Training model!')


#Choose number of epochs here
history = model.fit(train_inputs, y_train, epochs=50, batch_size=32, verbose=1)

print("Evaluating Keras model")
eval_results = model.evaluate(test_inputs, y_test, verbose=1)
print("Keras Model RMSE (log scale):", eval_results[1])


# Get predictions (on log scale).
y_pred_log = model.predict(test_inputs).flatten()
# Optionally, convert predictions back to original scale:
y_pred = np.exp(y_pred_log)
y_test_actual = np.exp(y_test)

# Compute R² on the log scale.
from sklearn.metrics import r2_score
r2_keras = r2_score(y_test, y_pred_log)
print("Keras Model R² (log scale):", r2_keras)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_log, color='blue', label='Keras Predicted (log math score)')
plt.scatter(y_test, y_test, color='red', label='Actual (log math score)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', label='Ideal')
plt.xlabel("Actual log(math score)")
plt.ylabel("Predicted log(math score)")
plt.title("Actual vs Predicted log(math score) for Keras Model")
plt.legend()
plt.show()



