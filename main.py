from emnist import extract_training_samples
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import os
import cv2
import numpy
import joblib


# # Grab the data from the OpenML website
# # X will be the images and y will be the labels
# X, y = extract_training_samples('letters')

# # Make sure that every pixel in all the images is a value between 0 and 1
# X = X / 255.

# # Use the first 60000 instances as training and the next 10000 as testing
# X_train, X_test = X[:60000], X[60000:70000]
# y_train, y_test = y[:60000], y[60000:70000]

# # record the number of samples in each dataset and the number of pixels in each image
# X_train = X_train.reshape(60000,784)
# X_test = X_test.reshape(10000,784)

# print("Extracted the samples and divided the training and testing data sets")



# img_index = 14000 # <<<<<  This value can be updated to look at other images
# img = X_train[img_index]
# print("Image Label: " + str(chr(y_train[img_index]+96)))
# plt.imshow(img.reshape((28,28)))
#
#

#this is the trained model
# model = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,), max_iter=50, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1)
# model.fit(X_train, y_train)
#
# # Save the trained model to a file
# joblib.dump(model, 'trained_model.joblib')

# Load the trained model from the file
loaded_model = joblib.load('trained_model.joblib')

# print("Training set score: %f" % model.score(X_train, y_train))
# print("Test set score: %f" % model.score(X_test, y_test))

# Load an individual image that you want to recognize
individual_image_path = 'letters_mod/02.jpg'
individual_image = cv2.imread(individual_image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the individual image
individual_image = cv2.GaussianBlur(individual_image, (7, 7), 0)
points = cv2.findNonZero(individual_image)
x, y, w, h = cv2.boundingRect(points)
if w > 0 and h > 0:
    if w > h:
        y = y - (w - h) // 2
        individual_image = individual_image[y:y + w, x:x + w]
    else:
        x = x - (h - w) // 2
        individual_image = individual_image[y:y + h, x:x + h]
individual_image = cv2.resize(individual_image, (28, 28), interpolation=cv2.INTER_CUBIC)
individual_image = individual_image / 255.0
individual_image_flat = individual_image.reshape(1, 784)

prediction = loaded_model.predict(individual_image_flat)
recognized_letter = chr(prediction[0] + 96)

print(f"AI: {recognized_letter}")

# Display the individual image and the recognized letter
plt.imshow(individual_image, cmap='gray')
plt.title(f"AI: {recognized_letter}")
plt.axis('off')
plt.show()

