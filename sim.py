import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import joblib

# Create synthetic data with RGB colors
np.random.seed(42)

# Generate random RGB values
data_size = 1000
colors = np.random.randint(0, 256, size=(data_size, 3))  # RGB values between 0 and 255

# Define a function to categorize color
def categorize_color(rgb):
    r, g, b = rgb
    brightness = (0.299*r + 0.587*g + 0.114*b)
    if brightness < 85:
        return 'dark'
    elif brightness > 170:
        return 'white'
    else:
        return 'colored'

# Generate labels based on color categories
labels = np.apply_along_axis(categorize_color, 1, colors)

# Create a DataFrame
data = pd.DataFrame(colors, columns=['red', 'green', 'blue'])
data['label'] = labels

# Encode labels to integers
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the data
X = data[['red', 'green', 'blue']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 3 output units for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Save the model to file
model.save('laundry_sorting_model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Predict new data
new_laundry_item = np.array([[30, 144, 255]])  # Example: RGB color
new_laundry_item_scaled = scaler.transform(new_laundry_item)

prediction = model.predict(new_laundry_item_scaled)
predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
print('Predicted label:', predicted_label[0])

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
