import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


### Multi-class


# Example data preparation
X_train = np.random.rand(100, 10, 1)  # 100 sequences, each of length 10, with 1 feature
y_train = np.random.randint(0, 3, 100)  # 100 labels for 3 classes
y_train = to_categorical(y_train, num_classes=3)  # One-hot encode the labels

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(10, 1)))  # LSTM layer with 50 units
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Example prediction
X_test = np.random.rand(1, 10, 1)  # Single test sequence
predictions = model.predict(X_test)



### Binary

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Example data preparation
X_train = np.random.rand(100, 10, 1)  # 100 sequences, each of length 10, with 1 feature
y_train = np.random.randint(0, 2, 100)  # 100 binary labels

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(10, 1)))  # LSTM layer with 50 units
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Example prediction
X_test = np.random.rand(1, 10, 1)  # Single test sequence
predictions = model.predict(X_test)

# The result will be a probability for the positive class
print("Predicted probability:", predictions)
