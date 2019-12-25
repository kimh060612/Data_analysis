import tensorflow as tf
import keras as K
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

data_path = "./Original_data/train/"
data_path_test = "./Original_data/test/"
data_list = os.listdir(data_path)
data_list_test = os.listdir(data_path_test)
Img_data = []
label = []

Y = []
onehot = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

for con in data_list:
    name = con.split(".")[0]
    label.append(name)

label = list(set(label))
print(label)

for con in data_list:
    name = con.split(".")[0]
    Y.append(onehot[label.index(name)])
    image_train = cv2.imread(data_path+con, cv2.IMREAD_GRAYSCALE)
    Img_data.append(image_train)

print(Img_data)

'''
model = K.models.Sequential()
model.add(K.layers.Conv2D(32, (3,3), activation="relu", input_shape=(300,300,1)))
model.add(K.layers.MaxPooling2D(2,2))
model.add(K.layers.Conv2D(64, (3,3), activation="relu"))
model.add(K.layers.MaxPooling2D(2,2))
model.add(K.layers.Conv2D(64, (3,3), activation="relu"))
model.add(K.layers.Flatten())
model.add(K.layers.Dense(64, activation='relu'))
model.add(K.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

print("Test Accuarcy:", test_acc*100, "%")
'''