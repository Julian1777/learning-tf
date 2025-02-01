import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('student_performance.csv')

print('Data Columns:', data.columns)
print(data.info())
print(data.describe())

# Split data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# Scale target variable to [0, 1] range
target_scaler = MinMaxScaler()
y_train = target_scaler.fit_transform(train_df[['math_score']]).flatten()
y_test = target_scaler.transform(test_df[['math_score']]).flatten()

# Remove target from features
train_df = train_df.drop(columns=['math_score'])
test_df = test_df.drop(columns=['math_score'])

categorical_columns = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
numeric_columns = ['reading_score', 'writing_score']

# Scale numeric features
feature_scaler = StandardScaler()
train_df[numeric_columns] = feature_scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = feature_scaler.transform(test_df[numeric_columns])

# Feature column setup
feature_columns = []
for feature_name in categorical_columns:
    vocabulary = train_df[feature_name].unique()
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
    feature_columns.append(tf.feature_column.indicator_column(cat_col))

for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Model architecture
inputs = {}
for col in categorical_columns:
    inputs[col] = keras.Input(shape=(1,), name=col, dtype=tf.string)
for col in numeric_columns:
    inputs[col] = keras.Input(shape=(1,), name=col, dtype=tf.float32)

feature_layer = keras.layers.DenseFeatures(feature_columns)
x = feature_layer(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.1)(x)
x = keras.layers.Dense(32, activation='relu')(x)
output = keras.layers.Dense(1, activation='sigmoid')(x)  # Constrained to 0-1

model = keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])

# Prepare input data
train_inputs = {name: np.array(value) for name, value in train_df.items()}
test_inputs = {name: np.array(value) for name, value in test_df.items()}

# Model training
print('Training model!')
history = model.fit(train_inputs, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluation
print("Evaluating Keras model")
eval_results = model.evaluate(test_inputs, y_test, verbose=1)
print("Keras Model RMSE (normalized scale):", eval_results[1])

# Predictions and inverse scaling
y_pred_normalized = model.predict(test_inputs).flatten()
y_pred = target_scaler.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
r2 = r2_score(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
print("\nKeras Model R² (original scale):", r2)
print("Keras Model RMSE (original scale):", rmse)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.6, label='Predictions')
plt.plot([0, 100], [0, 100], 'r--', label='Ideal')
plt.xlabel('Actual Math Score')
plt.ylabel('Predicted Math Score')
plt.title('Actual vs Predicted Math Scores')
plt.legend()
plt.grid(True)
plt.show()