# Exercise 3.4 of EECE680C Lecture 03
# implement 2-layer ANN to learn the XOR problem
# This program needs Tensorflow 2.*, not 1.*

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1. define training data
X = np.array([[0., 0., 1.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])  # input training array X
Y = np.array([[1.], [1.], [0.], [1.], [0.], [0.], [1.]])  # output training array Y

# Step 2. define DNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Step 3. Define loss and training parameters
predictions = model(X).numpy()
print(predictions)

loss_fn = tf.keras.losses.MeanSquaredError()
print(loss_fn(Y, predictions).numpy())

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Step 4. Train
history = model.fit(X, Y, epochs=1000)
print(history.history.keys())

# Step 5. Plot loss learning curve
plt.plot(history.history['loss'])
plt.title('Learning Curve')
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.show()

# Step 6. Plot accuracy curve
plt.plot(history.history['accuracy'])
plt.title('Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()