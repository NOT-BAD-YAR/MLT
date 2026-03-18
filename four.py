import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np 
import matplotlib.pyplot as plt 
 
# Load MNIST dataset 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 

# Display dataset shapes 
print("Training data shape:", x_train.shape) 
print("Testing data shape:", x_test.shape) 
 
# Normalize the pixel values (0-255 to 0-1) 
x_train = x_train / 255.0 
x_test = x_test / 255.0 
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784) 
 
model = Sequential() 
model.add(Dense(128, activation='relu', input_shape=(784,))) 
model.add(Dense(64, activation='relu')) 
 
# Output Layer (10 classes) 
model.add(Dense(10, activation='softmax')) 
 
# Compile the model 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 
 
# Train the model using Backpropagation 
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2) 
 
# Evaluate the model 
test_loss, test_accuracy = model.evaluate(x_test, y_test) 
print("Test Loss:", test_loss) 
print("Test Accuracy:", test_accuracy) 
 
# Make predictions 
predictions = model.predict(x_test) 
 
# Display predicted and actual value 
predicted_label = np.argmax(predictions[0]) 
actual_label = y_test[0] 
print("Predicted Digit:", predicted_label) 
print("Actual Digit:", actual_label) 
 
# Display the image 
plt.imshow(x_test[0].reshape(28,28), cmap='gray') 
plt.title("Sample Test Image") 
plt.show()
