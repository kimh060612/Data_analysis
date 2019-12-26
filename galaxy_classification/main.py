import tensorflow as tf
import keras as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

data_path = "./Resize_data/train_galaxy/"
data_path_test = "./Resize_data/test_galaxy/"
data_list = os.listdir(data_path)
data_list_test = os.listdir(data_path_test)
Img_data = []
Img_data_test = []

dataGen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range= 0.2, rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode="nearest")

Y = []
Y_test = []
onehot = [[1,0,0],[0,1,0],[0,0,1]]

for i in range(len(data_list)):
    img_path = data_path + data_list[i] + "/"
    img_list = os.listdir(img_path)
    for j in range(len(img_list)):
        Y.append(onehot[i])
        image_train = cv2.imread(img_path + img_list[j], cv2.IMREAD_GRAYSCALE)
        Img_data.append(image_train.tolist())

for i in range(len(data_list_test)):
    img_path = data_path_test + data_list_test[i] + "/"
    img_list = os.listdir(img_path)
    for j in range(len(img_list)):
        Y_test.append(onehot[i])
        image_test = cv2.imread(img_path + img_list[j], cv2.IMREAD_GRAYSCALE)
        Img_data_test.append(image_test.tolist())

Y = np.array(Y).reshape((-1,3))
Img_data = np.array(Img_data).reshape((-1,50,50,1))

Img_data = dataGen.flow_from_dataframe(Img_data, target_size=(50,50), batch_size= 32, class_mode="binary")

Y_test = np.array(Y_test).reshape((-1,3))
Img_data_test = np.array(Img_data_test).reshape((-1,50,50,1))
print(Img_data.shape)


model = K.models.Sequential()
model.add(K.layers.Conv2D(32, (3,3), activation="relu", input_shape=(50,50,1)))
model.add(K.layers.MaxPooling2D(2,2))
model.add(K.layers.Conv2D(64, (3,3), activation="relu"))
model.add(K.layers.MaxPooling2D(2,2))
model.add(K.layers.Conv2D(64, (3,3), activation="relu"))
model.add(K.layers.MaxPooling2D(2,2))
model.add(K.layers.Conv2D(64, (3,3), activation="relu"))
model.add(K.layers.Flatten())
model.add(K.layers.Dropout(0.4))
model.add(K.layers.Dense(64, activation='relu'))
model.add(K.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(Img_data, Y, epochs=50)

test_loss, test_acc = model.evaluate(Img_data_test,  Y_test, verbose=2)

print(test_acc*100, "% ", "Finish")
