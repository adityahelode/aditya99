from IPython.display import Image, display
# preprocessing and processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
# ploting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.utils import plot_model
# split data
from sklearn.model_selection import train_test_split
# CNN
from keras import models, layers
# val
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import os

labels = os.listdir(r"C:\Users\adihe\OneDrive\Desktop\Python\Lab\natural_images")
print(labels)

num = []
for label in labels:
    path = r"C:\Users\adihe\OneDrive\Desktop\Python\Lab\natural_images/{0}/".format(label)
    folder_data = os.listdir(path)
    k = 0
    print('\n',f'=====   {label.upper()}   =====')
    for image_path in folder_data:
        if k < 5:
            display(Image(path+image_path))
        k = k+1
    num.append(k)
    print(f'count : {k} images , label : {label} class')

    x_data =[]
y_data = []
import cv2
for label in labels:
    path = r"C:\Users\adihe\OneDrive\Desktop\Python\Lab\natural_images/{0}/".format(label)
    folder_data = os.listdir(path)
    for image_path in folder_data:
        image = cv2.imread(path+image_path)
        image_resized = cv2.resize(image, (32,32))
        x_data.append(np.array(image_resized))
        y_data.append(label)
x_data = np.array(x_data)
y_data = np.array(y_data)
print('the shape of X is: ', x_data.shape, 'and that of Y is: ', y_data.shape)

#stadardizing the input data
x_data = x_data.astype('float32')/255

y_encoded = LabelEncoder().fit_transform(y_data)
y_categorical = to_categorical(y_encoded)

r = np.arange(x_data.shape[0])
np.random.seed(42)
np.random.shuffle(r)
X = x_data[r]
Y = y_categorical[r]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)

model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(8, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=5, validation_split=0.2)

import matplotlib.pyplot as plt

plt.figure(figsize = (4,4))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss/accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

Y_pred = np.argmax(model.predict(X_test), axis=1)

Y_test = np.argmax(Y_test, axis = 1)

accuracy_score(Y_pred,Y_test)

print(classification_report(Y_test, Y_pred))