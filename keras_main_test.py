import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ================================
# 1. Load the Data
# ================================
# We assume that 'car_train.csv' contains all 10,000 rows.
data = pd.read_csv('car_train.csv')

print("Data Columns:", data.columns)
print(data.describe())

# ================================
# 2. Create Train/Test Splits
# ================================
# Instead of using a separate evaluation file, we use all 10,000 rows
# and split them into training (80%) and testing (20%).
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# ================================
# 3. Transform the Target Variable
# ================================
# We use the natural logarithm of Price to reduce skewness.
# (Remember: after prediction, you can convert back with np.exp.)
y_train = np.log(train_df.pop('Price'))
y_test = np.log(test_df.pop('Price'))

# ================================
# 4. Preprocess the Features
# ================================
# Identify categorical and numeric columns.
categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
numeric_columns = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']

# Standardize the numeric features.
scaler = StandardScaler()
train_df[numeric_columns] = scaler.fit_transform(train_df[numeric_columns])
test_df[numeric_columns] = scaler.transform(test_df[numeric_columns])

# ================================
# 5. Build Feature Columns for the Keras Model
# ================================
# We use tf.feature_column to create inputs for our Keras model.
# For categorical features, we need to convert them into dense (one-hot) representations.
feature_columns = []
for feature_name in categorical_columns:
    vocabulary = train_df[feature_name].unique()
    # Create a categorical column with a vocabulary list.
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
    # Wrap it with indicator_column to get one-hot encoded tensors.
    ind_col = tf.feature_column.indicator_column(cat_col)
    feature_columns.append(ind_col)

# Add numeric columns directly.
for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# ================================
# 6. Build the Keras Model (Functional API)
# ================================
# Define separate inputs for categorical and numeric features.
inputs = {}
# Categorical inputs are strings.
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

# Create the model.
model = keras.Model(inputs=inputs, outputs=output)

# Compile the model.
# We use Mean Squared Error (mse) as the loss and track Root Mean Squared Error (RMSE).
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])

# ================================
# 7. Prepare Input Data for Keras
# ================================
# Keras expects a dictionary mapping feature names to arrays.
train_inputs = {name: np.array(value) for name, value in train_df.items()}
test_inputs = {name: np.array(value) for name, value in test_df.items()}

# ================================
# 8. Train the Keras Model
# ================================
print("Training Keras model...")
history = model.fit(train_inputs, y_train, epochs=50, batch_size=32, verbose=1)
# Increase epochs to 50 to allow more training time.

# Evaluate the model on the test set.
print("Evaluating Keras model...")
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

# ================================
# 9. Build and Evaluate RandomForestRegressor
# ================================
# For tree-based models, we cannot feed string features directly.
# We'll use one-hot encoding for the categorical features.
rf_train = pd.get_dummies(train_df, columns=categorical_columns)
rf_test = pd.get_dummies(test_df, columns=categorical_columns)
# Align the columns to ensure the same set appears in both.
rf_train, rf_test = rf_train.align(rf_test, join='left', axis=1, fill_value=0)

print("Training RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
rf_model.fit(rf_train, y_train)
y_pred_rf = rf_model.predict(rf_test)
r2_rf = r2_score(y_test, y_pred_rf)
print("RandomForestRegressor R² (log scale):", r2_rf)

# ================================
# 10. Visualize Predictions
# ================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, color='blue', label='RF Predicted (log Price)')
plt.scatter(y_test, y_test, color='red', label='Actual (log Price)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', label='Ideal')
plt.xlabel("Actual log(Price)")
plt.ylabel("Predicted log(Price)")
plt.title("Actual vs Predicted log(Price) for RandomForestRegressor")
plt.legend()
plt.show()
