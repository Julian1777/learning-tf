import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import feature_column as fc
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

tf.compat.v1.disable_eager_execution()

dftrain = pd.read_csv('car_train.csv')
dfeval = pd.read_csv('car_eval.csv')

print("Train Columns:", dftrain.columns)
print("Eval Columns:", dfeval.columns)
print(dftrain.describe())
print(dfeval.describe())

# Apply log transformation to Price
y_train = np.log(dftrain.pop('Price'))
y_eval = np.log(dfeval.pop('Price'))

categorical_columns = ['Brand', 'Model', 'Fuel_Type', 'Transmission']
numeric_columns = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']

# Normalize numeric features
scaler = StandardScaler()
dftrain[numeric_columns] = scaler.fit_transform(dftrain[numeric_columns])
dfeval[numeric_columns] = scaler.transform(dfeval[numeric_columns])

feature_columns = []
for feature_name in categorical_columns:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=25, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# DNNRegressor
print("Training DNNRegressor...")
dnn_est = tf.estimator.DNNRegressor(
    feature_columns=feature_columns,
    hidden_units=[64, 32],
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)
dnn_est.train(train_input_fn)
result = dnn_est.evaluate(eval_input_fn)
rmse = np.sqrt(result['average_loss'])
print("DNNRegressor RMSE:", rmse)

# RandomForestRegressor
print("Training RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(dftrain, y_train)
y_pred = rf_model.predict(dfeval)
r2 = r2_score(y_eval, y_pred)
print("RandomForest R² Score:", r2)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_eval, y_pred, color='blue', label='Predicted Prices')
plt.scatter(y_eval, y_eval, color='red', label='Actual Prices')
plt.plot([min(y_eval), max(y_eval)], [min(y_eval), max(y_eval)], color='green', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Prices (log scale)")
plt.ylabel("Predicted Prices (log scale)")
plt.title("Actual vs. Predicted Prices")
plt.legend()
plt.show()
